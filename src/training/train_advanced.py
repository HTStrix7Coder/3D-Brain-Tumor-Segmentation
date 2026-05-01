"""
BraTS 2023 - Advanced 3D U-Net Training Script
===============================================
Trains an advanced 3D U-Net with:
- Modality Gating
- Residual Blocks
- VAE Reconstruction Regularization

Optimized for RTX 4060 Ti 8GB.
"""
# C:\Users\harin\ai_lab\Scripts\activate.bat
# python src/training/train_advanced.py
import os
import sys
import time
import yaml 
import torch
import torch.nn as nn
import numpy as np
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.data_utils import get_dataloaders
from src.utils.metrics import get_metrics, get_post_transforms
from src.models.advanced_unet import AdvancedBraTSNet


def load_config(config_path="configs/advanced.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(
    model, train_loader, optimizer, dice_loss_fn, l2_loss_fn, 
    scaler, device, grad_accum_steps, epoch, vae_weight
):
    """
    Train for one epoch with VAE regularization.
    """
    model.train()
    epoch_loss = 0
    epoch_seg_loss = 0
    epoch_vae_loss = 0
    step_count = 0

    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(train_loader):
        inputs = batch_data["image"].to(device)
        labels = batch_data["seg"].to(device)

        with torch.amp.autocast("cuda"):
            # Model returns both segmentation and reconstruction
            seg_out, vae_out = model(inputs)
            
            # Segmentation Loss (Dice)
            seg_loss = dice_loss_fn(seg_out, labels)
            
            # VAE Reconstruction Loss (MSE/L2)
            vae_loss = l2_loss_fn(vae_out, inputs)
            
            # Combined Loss
            total_loss = seg_loss + (vae_weight * vae_loss)
            total_loss = total_loss / grad_accum_steps

        scaler.scale(total_loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += total_loss.item() * grad_accum_steps
        epoch_seg_loss += seg_loss.item()
        epoch_vae_loss += vae_loss.item()
        step_count += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Step {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {total_loss.item() * grad_accum_steps:.4f} "
                  f"(Seg: {seg_loss.item():.4f}, VAE: {vae_loss.item():.4f})")

    if (batch_idx + 1) % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return epoch_loss / step_count, epoch_seg_loss / step_count, epoch_vae_loss / step_count


from tqdm import tqdm

@torch.no_grad()
def validate(model, val_loader, dice_loss_fn, dice_metric, device, roi_size):
    """
    Validate on full volumes. In validation, VAE branch is usually disabled.
    """
    model.eval()
    post_sigmoid, post_pred = get_post_transforms()
    val_loss = 0
    count = 0

    print("  Validating...")
    for batch_data in tqdm(val_loader, desc="  Val Progress"):
        inputs = batch_data["image"].to(device)
        labels = batch_data["seg"].to(device)

        with torch.amp.autocast("cuda"):
            # In eval() mode, AdvancedBraTSNet only returns seg_out
            outputs = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=1, # Reduced for advanced model
                predictor=model,
                overlap=0.5,
            )
            loss = dice_loss_fn(outputs, labels)

        val_loss += loss.item()
        count += 1

        outputs_list = decollate_batch(outputs)
        labels_list = decollate_batch(labels)
        outputs_processed = [post_pred(post_sigmoid(x)) for x in outputs_list]
        dice_metric(y_pred=outputs_processed, y=labels_list)

    dice_scores = dice_metric.aggregate()
    dice_metric.reset()

    mean_val_loss = val_loss / count
    dice_tc = dice_scores[0].item()
    dice_wt = dice_scores[1].item()
    dice_et = dice_scores[2].item()
    mean_dice = (dice_wt + dice_tc + dice_et) / 3

    return mean_val_loss, dice_wt, dice_tc, dice_et, mean_dice


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("BraTS 2023 - Advanced U-Net Training (VAE + Gating)")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Create directories
    os.makedirs(config["data"]["results_dir"], exist_ok=True)
    os.makedirs(config["data"]["checkpoint_dir"], exist_ok=True)

    # Data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(config)

    # Model
    model = AdvancedBraTSNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"]["features"],
        vae_reg=config["model"]["vae_reg"]
    ).to(device)
    
    print(f"Model: {config['model']['name']}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Losses
    dice_loss_fn = DiceLoss(sigmoid=True, squared_pred=True, smooth_dr=1e-5)
    l2_loss_fn = nn.MSELoss()
    vae_weight = config["model"]["vae_weight"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["max_epochs"]
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config["optimization"]["amp"])
    dice_metric, _ = get_metrics()

    # Resume Logic
    start_epoch = 1
    best_dice = 0.0
    latest_path = os.path.join(config["data"]["checkpoint_dir"], "latest_advanced.pth")
    if os.path.exists(latest_path):
        print(f"Resuming from {latest_path}...")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint["best_dice"]

    # Training log
    log_path = os.path.join(config["data"]["results_dir"], "advanced_training_log.csv")
    if start_epoch == 1:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,dice_wt,dice_tc,dice_et,mean_dice,lr\n")

    print("\nStarting training...")
    for epoch in range(start_epoch, config["training"]["max_epochs"] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, seg_loss, vae_loss = train_one_epoch(
            model, train_loader, optimizer, dice_loss_fn, l2_loss_fn,
            scaler, device, config["training"]["grad_accumulation"], 
            epoch, vae_weight
        )
        scheduler.step()
        
        # Validate
        if epoch % config["training"]["val_interval"] == 0:
            val_loss, dice_wt, dice_tc, dice_et, mean_dice = validate(
                model, val_loader, dice_loss_fn, dice_metric, device, config["patch"]["roi_size"]
            )
            
            print(f"\nEpoch {epoch} | Loss: {train_loss:.4f} (Seg: {seg_loss:.4f}, VAE: {vae_loss:.4f}) | Val Dice: {mean_dice:.4f}")
            print(f"  WT: {dice_wt:.4f} | TC: {dice_tc:.4f} | ET: {dice_et:.4f}")
            
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_dice": best_dice,
                }, os.path.join(config["data"]["checkpoint_dir"], "best_advanced.pth"))
                print("  ★ New best model saved!")

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{dice_wt:.6f},{dice_tc:.6f},{dice_et:.6f},{mean_dice:.6f},{scheduler.get_last_lr()[0]:.8f}\n")
        else:
            print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Time: {time.time()-epoch_start:.0f}s")

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": best_dice,
        }, latest_path)

if __name__ == "__main__":
    main()
