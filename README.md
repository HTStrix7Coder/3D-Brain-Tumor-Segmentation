# NeuroGate-VAE: Uncertainty-Aware Brain Tumor Segmentation
### BraTS 2023 | 3D Hybrid Transformer + VAE Regularization

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![MONAI](https://img.shields.io/badge/MONAI-1.2+-green)

## 🧠 Project Overview
NeuroGate-VAE is an advanced medical imaging pipeline designed for the **BraTS 2023 (Glioma)** challenge. It moves beyond standard 3D U-Nets by integrating **Adaptive Modality Gating** and **Global Self-Attention** to segment brain tumors from multi-modal MRI scans (T1, T1c, T2, FLAIR). 

The project's standout feature is its **Uncertainty Estimation Engine**, which uses a VAE reconstruction branch to identify regions where the model is clinically uncertain.

## 🚀 Key Results
| Region | Dice Score |
| :--- | :--- |
| **Whole Tumor (WT)** | **0.9034** |
| **Tumor Core (TC)** | **0.8662** |
| **Enhancing Tumor (ET)** | **0.8529** |
| **Mean Dice** | **0.8742** |

## 🏗️ Architecture
The **NeuroGate-VAE** architecture consists of:
1.  **Adaptive Modality Gating:** Learns per-patient weights for the 4 MRI modalities, automatically prioritizing the most informative scans (e.g., T1c for the enhancing core).
2.  **Hybrid Encoder:** Residual CNN blocks combined with a **Global Self-Attention** bottleneck to capture both local tumor edges and global brain symmetry.
3.  **VAE Regularization Branch:** A variational autoencoder that reconstructs the input brain during training. This forces the model to learn the underlying 3D anatomy, not just pixels.
4.  **Uncertainty Engine:** During inference, the VAE reconstruction error is used to generate voxel-wise uncertainty maps, providing a "confidence score" for every segmented pixel.

## 📁 Project Structure
```bash
├── configs/          # Hyperparameters and data paths
├── data/             # BraTS 2023 Dataset (T1, T1n, T2f, T2w)
├── src/
│   ├── models/       # AdvancedBraTSNet implementation
│   ├── training/     # Training loop with AMP and Gradient Accumulation
│   ├── inference/    # Predict uncertainty and segmentation
│   └── utils/        # Data loading, metrics, and visualization
└── results/          # Model checkpoints and prediction reports
```

## 🛠️ Usage
### 1. Training
```bash
python src/training/train_advanced.py
```
### 2. Inference & Uncertainty
```bash
python src/inference/predict_uncertainty.py <patient_id>
```
### 3. Visualization
```bash
python src/utils/visualize_results.py <patient_id>
```

## 📊 Visualizations
The pipeline generates a 3-panel clinical report:
*   **Panel 1:** Original MRI (T1c).
*   **Panel 2:** Multi-region segmentation overlay (WT, TC, ET).
*   **Panel 3:** VAE Uncertainty "Lava" Map—highlighting areas of low model confidence.

## 🛡️ Developed By
**NeuroGate-VAE** was developed as a research-focused implementation of advanced 3D medical segmentation.

---
*Optimized for RTX 4060 Ti (8GB VRAM) using MONAI and PyTorch.*
