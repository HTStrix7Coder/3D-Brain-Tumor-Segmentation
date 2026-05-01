"""
BraTS 2023 Data Utilities
=========================
Handles data loading, preprocessing, and augmentation using MONAI.
BraTS 2023 GLI naming convention: t1c, t1n, t2f, t2w, seg
BraTS label mapping:
  0 = background
  1 = NCR/NET (Necrotic / Non-Enhancing Tumor Core)
  2 = ED (Peritumoral Edema)
  3 = ET (Enhancing Tumor)

Evaluation regions (multi-label):
  WT (Whole Tumor)     = labels 1 + 2 + 3
  TC (Tumor Core)      = labels 1 + 3
  ET (Enhancing Tumor) = label 3
"""

import os
import torch
import glob
import numpy as np
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ConcatItemsd,
    MapTransform, # Added
    SpatialPadd,
    DivisiblePadd,
)
from monai.data import CacheDataset, DataLoader, decollate_batch, MetaTensor

class ConvertToMultiChannelBrats2023d(MapTransform):
    """
    Convert BraTS 2023 labels to multi-channel based on the hierarchy:
    Label 1: NCR/NET, Label 2: ED, Label 3: ET
    
    WT (Whole Tumor) = 1 + 2 + 3
    TC (Tumor Core)  = 1 + 3
    ET (Enhancing Tumor) = 3
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # Squeeze out the existing channel dimension (1, H, W, D) -> (H, W, D)
            label_map = d[key][0] if d[key].shape[0] == 1 else d[key]
            
            # TC (Tumor Core)
            result.append(torch.logical_or(label_map == 1, label_map == 3))
            # WT (Whole Tumor)
            result.append(
                torch.logical_or(
                    torch.logical_or(label_map == 1, label_map == 2), label_map == 3
                )
            )
            # ET (Enhancing Tumor)
            result.append(label_map == 3)
            
            # Stack and preserve MetaTensor metadata
            new_tensor = torch.stack(result, axis=0).float()
            if isinstance(d[key], MetaTensor):
                d[key] = MetaTensor(new_tensor, affine=d[key].affine, meta=d[key].meta)
            else:
                d[key] = new_tensor
        return d


def get_patient_list(data_dir):
    """
    Scan the data directory and build a list of patient dictionaries.
    Each dict has keys: t1c, t1n, t2f, t2w, seg (file paths).
    """
    patients = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("BraTS")
    ])

    data_dicts = []
    for patient_id in patients:
        patient_dir = os.path.join(data_dir, patient_id)
        entry = {
            "t1c": os.path.join(patient_dir, f"{patient_id}-t1c.nii.gz"),
            "t1n": os.path.join(patient_dir, f"{patient_id}-t1n.nii.gz"),
            "t2f": os.path.join(patient_dir, f"{patient_id}-t2f.nii.gz"),
            "t2w": os.path.join(patient_dir, f"{patient_id}-t2w.nii.gz"),
            "seg": os.path.join(patient_dir, f"{patient_id}-seg.nii.gz"),
        }

        # Verify all files exist
        if all(os.path.exists(v) for v in entry.values()):
            data_dicts.append(entry)
        else:
            missing = [k for k, v in entry.items() if not os.path.exists(v)]
            print(f"  WARNING: Skipping {patient_id}, missing: {missing}")

    return data_dicts


def get_train_val_split(data_dir, val_ratio=0.2, seed=42):
    """
    Split the training data into train and validation sets.
    Returns (train_dicts, val_dicts).
    """
    all_dicts = get_patient_list(data_dir)
    print(f"Found {len(all_dicts)} valid patients in {data_dir}")

    train_dicts, val_dicts = train_test_split(
        all_dicts, test_size=val_ratio, random_state=seed
    )
    print(f"  Train: {len(train_dicts)} | Val: {len(val_dicts)}")
    return train_dicts, val_dicts


def get_train_transforms(roi_size=(128, 128, 128)):
    """
    Training transforms:
    1. Load NIfTI files
    2. Ensure channel-first format
    3. Orientation to RAS
    4. Normalize each modality independently (z-score)
    5. Convert BraTS labels to multi-channel (WT, TC, ET)
    6. Crop foreground (remove empty space around brain)
    7. Random spatial crop to roi_size
    8. Data augmentation (flip, rotate, intensity)
    """
    modality_keys = ["t1c", "t1n", "t2f", "t2w"]
    all_keys = modality_keys + ["seg"]

    return Compose([
        # Load
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),

        # 1. Spatial Operations (Perform while everything is 1-channel)
        CropForegroundd(keys=all_keys, source_key="t1c", margin=10),
        SpatialPadd(keys=all_keys, spatial_size=roi_size),

        # 2. Intensity Operations
        NormalizeIntensityd(keys=modality_keys, nonzero=True, channel_wise=True),

        # 3. Channel Operations (Stack modalities and convert labels)
        ConcatItemsd(keys=modality_keys, name="image", dim=0),
        ConvertToMultiChannelBrats2023d(keys="seg"),

        # 4. Patching & Augmentation (Now safe to use image and seg together)
        RandSpatialCropd(
            keys=["image", "seg"],
            roi_size=roi_size,
            random_size=False,
        ),

        # Augmentation
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "seg"], prob=0.3, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),

        EnsureTyped(keys=["image", "seg"], dtype=torch.float32),
    ])


def get_val_transforms():
    """
    Validation transforms: no augmentation, no random cropping.
    We process entire volumes for accurate evaluation.
    """
    modality_keys = ["t1c", "t1n", "t2f", "t2w"]
    all_keys = modality_keys + ["seg"]

    return Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        CropForegroundd(keys=all_keys, source_key="t1c", margin=5),
        
        # DivisiblePad while still 1-channel
        DivisiblePadd(keys=all_keys, k=8),
        
        NormalizeIntensityd(keys=modality_keys, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=modality_keys, name="image", dim=0),
        ConvertToMultiChannelBrats2023d(keys="seg"),
        EnsureTyped(keys=["image", "seg"], dtype=torch.float32),
    ])


def get_dataloaders(config):
    """
    Build train and validation DataLoaders from config dict.
    Uses CacheDataset for faster training (caches cache_rate % in RAM).
    """
    train_dicts, val_dicts = get_train_val_split(
        config["data"]["train_dir"],
        val_ratio=0.2,
        seed=config["training"]["seed"],
    )

    roi_size = config["patch"]["roi_size"]
    cache_rate = config["optimization"]["cache_rate"]
    num_workers = config["training"]["num_workers"]
    batch_size = config["training"]["batch_size"]

    train_ds = CacheDataset(
        data=train_dicts,
        transform=get_train_transforms(roi_size=roi_size),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    val_ds = CacheDataset(
        data=val_dicts,
        transform=get_val_transforms(),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Full volume validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
