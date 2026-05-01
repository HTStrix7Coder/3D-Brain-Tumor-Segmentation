import os
import sys
import yaml
import torch
import numpy as np
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.advanced_unet import AdvancedBraTSNet

def predict_with_uncertainty(patient_id, config_path="configs/advanced.yaml"):
    # 1. Load Config and Model
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AdvancedBraTSNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"]["features"],
        vae_reg=True
    ).to(device)
    
    # Load best checkpoint
    checkpoint_path = os.path.join(config["data"]["checkpoint_dir"], "best_advanced.pth")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.eval()
    
    # 2. Find Patient Data
    train_dir = config["data"]["train_dir"]
    val_dir = config["data"]["val_dir"]
    
    if os.path.exists(os.path.join(val_dir, patient_id)):
        patient_dir = os.path.join(val_dir, patient_id)
    elif os.path.exists(os.path.join(train_dir, patient_id)):
        patient_dir = os.path.join(train_dir, patient_id)
    else:
        print(f"ERROR: Patient {patient_id} not found.")
        return

    output_dir = os.path.join("results", "predictions", patient_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Preprocess
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, NormalizeIntensityd, ConcatItemsd, EnsureTyped, DivisiblePadd
    
    modality_keys = ["t1c", "t1n", "t2f", "t2w"]
    data_dict = {k: os.path.join(patient_dir, f"{patient_id}-{k}.nii.gz") for k in modality_keys}
    
    seg_path = os.path.join(patient_dir, f"{patient_id}-seg.nii.gz")
    has_seg = os.path.exists(seg_path)
    if has_seg: data_dict["seg"] = seg_path
    
    pre_transforms = Compose([
        LoadImaged(keys=modality_keys + (["seg"] if has_seg else [])),
        EnsureChannelFirstd(keys=modality_keys + (["seg"] if has_seg else [])),
        Orientationd(keys=modality_keys + (["seg"] if has_seg else []), axcodes="RAS"),
        NormalizeIntensityd(keys=modality_keys, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=modality_keys, name="image", dim=0),
        DivisiblePadd(keys=["image"] + (["seg"] if has_seg else []), k=8),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    
    batch_data = pre_transforms(data_dict)
    inputs = batch_data["image"].unsqueeze(0).to(device)
    
    # 4. Inference
    print(f"  Running inference for {patient_id}...")
    roi_size = config["patch"]["roi_size"]
    
    with torch.no_grad():
        # --- EXPLAINABLE AI: Get Modality Weights ---
        _, weights = model(inputs, return_gating=True)
        weights_np = weights.cpu().numpy()[0]
        
        with open(os.path.join(output_dir, "modality_weights.txt"), "w") as f:
            modalities = ["T1-Contrast", "T1-Native", "T2-FLAIR", "T2-Weighted"]
            for name, w in zip(modalities, weights_np):
                f.write(f"{name}: {w:.4f}\n")
        
        # Get Segmentation
        seg_out = sliding_window_inference(inputs, roi_size, 4, model)
        seg_out = torch.sigmoid(seg_out) > 0.5
        
        # Get VAE Reconstruction (to calculate uncertainty)
        def vae_wrapper(x):
            x_gated = model.gating(x)
            e1 = model.enc1(x_gated)
            e2 = model.enc2(model.down1(e1))
            e3 = model.enc3(model.down2(e2))
            bottleneck = model.bottleneck_attention(model.down3(e3))
            b, c, h, w, d = bottleneck.shape
            x_flat = bottleneck.view(b, c, h * w * d).permute(0, 2, 1)
            x_attn, _ = model.attn(x_flat, x_flat, x_flat)
            bottleneck = x_attn.permute(0, 2, 1).view(b, c, h, w, d)
            return model.vae(bottleneck)

        recon_out = sliding_window_inference(inputs, roi_size, 4, vae_wrapper)
        
    # 5. Calculate Uncertainty
    uncertainty = torch.mean((inputs - recon_out)**2, dim=1, keepdim=True)
    
    # 6. Save Results
    print(f"  Saving results to {output_dir}...")
    affine = batch_data["image"].meta["affine"]
    seg_np = seg_out[0].cpu().numpy().astype(np.uint8)
    for i, name in enumerate(["TC", "WT", "ET"]):
        nib.save(nib.Nifti1Image(seg_np[i], affine), os.path.join(output_dir, f"pred_{name}.nii.gz"))
        
    unc_np = uncertainty[0, 0].cpu().numpy()
    unc_np = (unc_np - unc_np.min()) / (unc_np.max() - unc_np.min() + 1e-8)
    nib.save(nib.Nifti1Image(unc_np, affine), os.path.join(output_dir, "uncertainty_map.nii.gz"))
    
    print(f"✅ Analysis complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_with_uncertainty(sys.argv[1])
    else:
        predict_with_uncertainty("BraTS-GLI-00001-000")
