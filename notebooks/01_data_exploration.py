import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
)
from monai.utils import first

# Path configuration
DATA_ROOT = r"D:\AICode\Medical Lab\BraTS\data\train\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
RESULTS_DIR = r"D:\AICode\Medical Lab\BraTS\results"

def explore_sample(patient_id):
    patient_dir = os.path.join(DATA_ROOT, patient_id)
    
    # Define file paths based on BraTS 2023 naming convention
    data_dict = {
        "t1c": os.path.join(patient_dir, f"{patient_id}-t1c.nii.gz"),
        "t1n": os.path.join(patient_dir, f"{patient_id}-t1n.nii.gz"),
        "t2f": os.path.join(patient_dir, f"{patient_id}-t2f.nii.gz"),
        "t2w": os.path.join(patient_dir, f"{patient_id}-t2w.nii.gz"),
        "seg": os.path.join(patient_dir, f"{patient_id}-seg.nii.gz")
    }
    
    # Verify files exist
    for k, v in data_dict.items():
        if not os.path.exists(v):
            print(f"Error: Missing file {v}")
            return

    print(f"--- Exploring Patient: {patient_id} ---")
    
    # Loader using MONAI
    loader = Compose([
        LoadImaged(keys=["t1c", "t1n", "t2f", "t2w", "seg"]),
        EnsureChannelFirstd(keys=["t1c", "t1n", "t2f", "t2w", "seg"]),
        # Orientationd and Spacingd are standard in BraTS, but safe to verify
        Orientationd(keys=["t1c", "t1n", "t2f", "t2w", "seg"], axcodes="RAS"),
    ])
    
    data = loader(data_dict)
    
    print(f"Image shape: {data['t1c'].shape}")
    print(f"Label shape: {data['seg'].shape}")
    print(f"Label unique values: {np.unique(data['seg'])}")
    
    # Visualization
    slice_idx = data['t1c'].shape[3] // 2 # Middle slice
    
    plt.figure(figsize=(15, 5))
    
    modalities = ['t1c', 't1n', 't2f', 't2w', 'seg']
    titles = ['T1c (Contrast)', 'T1n (Native)', 'T2-FLAIR', 'T2-Weighted', 'Segmentation']
    
    for i, (mod, title) in enumerate(zip(modalities, titles)):
        plt.subplot(1, 5, i+1)
        img_slice = data[mod][0, :, :, slice_idx]
        
        if mod == 'seg':
            plt.imshow(img_slice, cmap='jet')
        else:
            plt.imshow(img_slice, cmap='gray')
            
        plt.title(title)
        plt.axis('off')
    
    output_path = os.path.join(RESULTS_DIR, f"exploration_{patient_id}.png")
    plt.savefig(output_path)
    print(f"Saved visualization to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Get first patient ID
    patients = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    if patients:
        explore_sample(patients[0])
    else:
        print("No patient folders found.")
