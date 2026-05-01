import nibabel as nib
import numpy as np
import os

# Check a random patient from your training data
data_dir = "D:/AICode/Medical Lab/BraTS/data/train/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
patient_id = "BraTS-GLI-00000-000"
seg_path = os.path.join(data_dir, patient_id, f"{patient_id}-seg.nii.gz")

if os.path.exists(seg_path):
    img = nib.load(seg_path)
    data = img.get_fdata()
    unique_labels = np.unique(data)
    print(f"Unique labels found in {patient_id}: {unique_labels}")
else:
    print(f"File not found: {seg_path}")
