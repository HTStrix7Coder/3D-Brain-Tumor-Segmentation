"""
BraTS 2023 - Baseline 3D U-Net Training Script
===============================================
Trains a simple 3D U-Net with early fusion (4 MRI channels concatenated)
on BraTS 2023 GLI dataset.

Optimized for RTX 4060 Ti 8GB:
  - Mixed precision training (AMP)
  - Gradient accumulation (effective batch size 4)
  - Patch-based training (128x128x128)
  - CacheDataset with low cache_rate

Usage:
  python src/training/train_baseline.py
"""
# C:\Users\harin\ai_lab\Scripts\activate.bat

import os
import sys
import time
import yaml
import torch
import numpy as np
from datetime import datetime

# Suppress TF logs from MONAI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.data_utils import get_dataloaders
from src.utils.metrics import get_metrics, get_post_transforms


def load_config(config_path="configs/default.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """
    Create a 3D U-Net model for BraTS segmentation.
    Input: 4 channels (T1n, T1c, T2w, T2-FLAIR)
    Output: 3 channels (WT, TC, ET) - multi-label sigmoid
    """
    model = UNet(
        spatial_dims=3,
        in_channels=config["model"]["in_channels"],     # 4
        out_channels=config["model"]["out_channels"],    # 3
        channels=config["model"]["features"],            # [32, 64, 128, 256]
        strides=(2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=0.1,
    )
    return model


def train_one_epoch(
    model, train_loader, optimizer, loss_fn, scaler,
    device, grad_accum_steps, epoch
):
    """
    Train for one epoch with AMP and gradient accumulation.
    """
    model.train()
    epoch_loss = 0
    step_count = 0

    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(train_loader):
        inputs = batch_data["image"].to(device)
        labels = batch_data["seg"].to(device)

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss = loss / grad_accum_steps  # Scale loss for accumulation

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * grad_accum_steps
        step_count += 1

        # Print progress every 50 steps
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Step {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item() * grad_accum_steps:.4f}")

    # Handle remaining gradients
    if (batch_idx + 1) % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return epoch_loss / step_count


@torch.no_grad()
def validate(model, val_loader, loss_fn, dice_metric, device, roi_size):
    """
    Validate on full volumes using sliding window inference.
    Returns mean Dice for WT, TC, ET.
    """
    model.eval()
    post_sigmoid, post_pred = get_post_transforms()
    val_loss = 0
    count = 0

    for batch_data in val_loader:
        inputs = batch_data["image"].to(device)
        labels = batch_data["seg"].to(device)

        # Sliding window inference for full-volume prediction
        with torch.amp.autocast("cuda"):
            outputs = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=2,
                predictor=model,
                overlap=0.5,
            )
            loss = loss_fn(outputs, labels)

        val_loss += loss.item()
        count += 1

        # Post-process predictions
        outputs_list = decollate_batch(outputs)
        labels_list = decollate_batch(labels)

        outputs_processed = [post_pred(post_sigmoid(x)) for x in outputs_list]
        dice_metric(y_pred=outputs_processed, y=labels_list)

    # Compute final Dice scores
    dice_scores = dice_metric.aggregate()
    dice_metric.reset()

    mean_val_loss = val_loss / count
    
    # MONAI Brats indices: 0:TC, 1:WT, 2:ET
    dice_tc = dice_scores[0].item()
    dice_wt = dice_scores[1].item()
    dice_et = dice_scores[2].item()
    
    mean_dice = (dice_wt + dice_tc + dice_et) / 3

    return mean_val_loss, dice_wt, dice_tc, dice_et, mean_dice


def main():
    # =========================================================================
    # Setup
    # =========================================================================
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("BraTS 2023 - Baseline 3D U-Net Training")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"AMP: {config['optimization']['amp']}")
    print(f"Grad Accumulation: {config['training']['grad_accumulation']}")
    print(f"Effective Batch Size: {config['training']['batch_size'] * config['training']['grad_accumulation']}")
    print(f"Patch Size: {config['patch']['roi_size']}")
    print()

    # Set seed for reproducibility
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    # Create output directories
    os.makedirs(config["data"]["results_dir"], exist_ok=True)
    os.makedirs(config["data"]["checkpoint_dir"], exist_ok=True)

    # =========================================================================
    # Data
    # =========================================================================
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print()

    # =========================================================================
    # Model, Loss, Optimizer
    # =========================================================================
    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: 3D U-Net")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # DiceLoss with sigmoid (multi-label, not softmax)
    loss_fn = DiceLoss(
        sigmoid=True,
        squared_pred=True,
        smooth_nr=0,
        smooth_dr=1e-5,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Learning rate scheduler - Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["max_epochs"],
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=config["optimization"]["amp"])

    # Metrics
    dice_metric, _ = get_metrics()

    # =========================================================================
    # Training Loop
    # =========================================================================
    best_dice = 0.0
    best_epoch = 0
    roi_size = config["patch"]["roi_size"]
    grad_accum = config["training"]["grad_accumulation"]
    val_interval = config["training"]["val_interval"]
    max_epochs = config["training"]["max_epochs"]

    # Training log
    log_path = os.path.join(config["data"]["results_dir"], "training_log.csv")
    
    # =========================================================================
    # Resume Logic
    # =========================================================================
    start_epoch = 1
    latest_checkpoint_path = os.path.join(
        config["data"]["checkpoint_dir"], "latest_checkpoint.pth"
    )
    
    if os.path.exists(latest_checkpoint_path):
        print(f"Loading latest checkpoint from {latest_checkpoint_path}...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint["best_dice"]
        print(f"  Resuming from epoch {start_epoch} (Best Dice so far: {best_dice:.4f})")
        
        # Open log in append mode
        log_file_mode = "a"
    else:
        # Start fresh
        log_file_mode = "w"
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,dice_wt,dice_tc,dice_et,mean_dice,lr\n")

    print()
    print("Starting training...")
    print("=" * 60)

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.time()

        # ----- Train -----
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler,
            device, grad_accum, epoch
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # ----- Validate -----
        if epoch % val_interval == 0:
            val_start = time.time()
            val_loss, dice_wt, dice_tc, dice_et, mean_dice = validate(
                model, val_loader, loss_fn, dice_metric, device, roi_size
            )
            val_time = time.time() - val_start

            print(f"\nEpoch {epoch}/{max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.0f}s + {val_time:.0f}s")
            print(f"  Dice WT: {dice_wt:.4f} | TC: {dice_tc:.4f} | ET: {dice_et:.4f} | "
                  f"Mean: {mean_dice:.4f} | LR: {current_lr:.6f}")

            # Save best model
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_epoch = epoch
                checkpoint_path = os.path.join(
                    config["data"]["checkpoint_dir"], "best_baseline_unet.pth"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_dice": best_dice,
                    "dice_wt": dice_wt,
                    "dice_tc": dice_tc,
                    "dice_et": dice_et,
                }, checkpoint_path)
                print(f"  ★ New best model saved! Mean Dice: {best_dice:.4f}")

            # Log to CSV
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                        f"{dice_wt:.6f},{dice_tc:.6f},{dice_et:.6f},"
                        f"{mean_dice:.6f},{current_lr:.8f}\n")
        else:
            print(f"Epoch {epoch}/{max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Time: {epoch_time:.0f}s | LR: {current_lr:.6f}")
        
        # ----- Save Latest Checkpoint -----
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": best_dice,
        }, latest_checkpoint_path)

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("Training Complete!")
    print(f"Best Mean Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"Training log saved to: {log_path}")
    print(f"Best model saved to: {config['data']['checkpoint_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
