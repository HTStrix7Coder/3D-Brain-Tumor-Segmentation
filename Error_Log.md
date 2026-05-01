# BraTS Project: Debugging Log & Lessons Learned

This file documents the technical hurdles we encountered during the initial setup and training of the 3D U-Net baseline on Windows.

## 1. Data Path & Nested Folders
- **Error:** `Missing file: ...`
- **Cause:** The downloaded BraTS dataset had a "double-nested" structure (e.g., `.../TrainingData/TrainingData/PatientID`).
- **Fix:** Updated `DATA_ROOT` in the config/script to point to the deepest folder containing the actual patient directories.

## 2. Spelling Typos
- **Error:** `ImportError: cannot import name 'HausdorfDistanceMetric' from 'monai.metrics'`
- **Cause:** A simple spelling mistake in the code (`Hausdorf` instead of `Hausdorff`).
- **Fix:** Corrected the spelling to `HausdorffDistanceMetric`.

## 3. Windows Shared Memory (Error 1455)
- **Error:** `RuntimeError: Couldn't open shared file mapping... error code: <1455>`
- **Cause:** PyTorch's `DataLoader` uses multi-processing on Windows, which requires a large "paging file" (virtual memory).
- **Fix:** Set `num_workers: 0` in the configuration. This runs data loading in the main process, which is more stable on Windows.

## 4. RAM Management (MemoryError)
- **Error:** `MemoryError` during dataset loading.
- **Cause:** We set `cache_rate: 0.2` (20%), which tried to cram ~45GB of raw MRI data into your 32GB of RAM.
- **Fix:** Reduced `cache_rate` to `0.1` (and later monitored it). For 32GB RAM, `0.05` to `0.1` is the upper limit for this specific dataset.

## 5. Tensor Type Mismatch (Bool vs. Float)
- **Error:** `RuntimeError: linalg.vector_norm: Expected a floating point... Got Bool`
- **Cause:** Newer versions of PyTorch/MONAI cannot calculate "norms" or "distances" on True/False (Boolean) masks.
- **Fix:** Added `.float()` casting to both the ground truth labels in the data loader and the model's predictions in the metrics script.

## 6. U-Net Dimension Mismatch (31 vs. 32)
- **Error:** `RuntimeError: Sizes of tensors must match except in dimension 1.`
- **Cause:** The 3D U-Net shrinks the image by factor of 2 at each level. If the input isn't a multiple of 8, the "skip connections" won't align (e.g., an image of 155 slices gets shrunk to 77, then 38, then 19... which doesn't perfectly multiply back up to 155).
- **Fix:** Added `SpatialPadd` (to ensure patches are at least 128) and `DivisiblePadd(k=8)` (to ensure full volumes are multiples of 8).

## 7. Metric Index Error
- **Error:** `IndexError: index 2 is out of bounds for dimension 0 with size 2`
- **Cause:** The `DiceMetric` was set to `include_background=False`. It treated the first tumor region as "background" and threw it away, leaving only 2 channels. We tried to access the 3rd channel, causing a crash.
- **Fix:** Set `include_background=True` because all 3 channels in the BraTS multi-channel format are important tumor regions (TC, WT, ET).

## 8. MONAI Block Import Error
- **Error:** `ImportError: cannot import name 'SwinTransformerBlock' from 'monai.networks.blocks'`
- **Cause:** In some versions of MONAI, the `SwinTransformerBlock` is located specifically within the `swin_unetr` module rather than the general `blocks` module.
- **Fix:** Changed the import to `from monai.networks.nets.swin_unetr import SwinTransformerBlock`.

## 9. MONAI ResidualUnit API Change
- **Error:** `TypeError: ResidualUnit.__init__() got an unexpected keyword argument 'nm_units'`
- **Cause:** In recent versions of MONAI, the argument to control the number of layers in a `ResidualUnit` was renamed from `nm_units` to `subunits`.
- **Fix:** Replaced all instances of `nm_units` with `subunits`.

## 10. SwinTransformerBlock API Change
- **Error:** `TypeError: SwinTransformerBlock.__init__() got an unexpected keyword argument 'input_resolution'`
- **Cause:** `SwinTransformerBlock` in MONAI handles resolution dynamically during the forward pass and does not require it in the constructor.
- **Fix:** Removed the `input_resolution` argument from the constructor.

## 11. SwinTransformerBlock Tuple Requirement
- **Error:** `TypeError: object of type 'int' has no len()` during Swin initialization.
- **Cause:** For 3D volumes, MONAI expects `window_size` and `shift_size` to be sequences (tuples) of 3 integers rather than a single integer.
- **Fix:** Converted `window_size=7` to `window_size=(7, 7, 7)` and `shift_size=3` to `shift_size=(3, 3, 3)`.

## 12. SwinTransformerBlock Masking Complexity
- **Error:** `TypeError: SwinTransformerBlock.forward() missing 1 required positional argument: 'mask_matrix'`
- **Cause:** Standalone `SwinTransformerBlock` usage requires manual computation of shifted window masks, which is highly complex for custom architectures.
- **Fix:** Replaced `SwinTransformerBlock` with a `MultiheadAttention` layer at the bottleneck. Since the bottleneck resolution is low (12x12x12), global attention is feasible and more robust than windowed attention.

## 13. BraTS 2023 Label Mismatch (Silent Failure)
- **Error:** `ET: 0.0000` plateauing indefinitely.
- **Cause:** BraTS 2023 uses label **3** for ET, but the MONAI `ConvertToMultiChannelBasedOnBratsClassesd` transform expects label **4**. This caused the ET channel to be consistently empty during training.
- **Fix:** Implemented a custom `ConvertToMultiChannelBrats2023d` transform in `data_utils.py` that explicitly maps labels `[1, 2, 3]` to the correct tumor sub-regions.

## 14. MetaTensor Metadata Loss
- **Error:** `ValueError: operands could not be broadcast together with shapes (3,) (4,)` during `CropForegroundd`.
- **Cause:** Creating a new tensor via `torch.stack` in a custom transform drops the MONAI `MetaTensor` metadata (affine, spacing, etc.). Subsequent spatial transforms like `CropForegroundd` fail because they cannot align the image and the mask.
- **Fix:** Moved **all spatial transforms** (`CropForegroundd`, `SpatialPadd`, `DivisiblePadd`) to the very beginning of the pipeline (before concatenation). This ensures all spatial operations occur while images are still 1-channel, preventing any broadcasting or metadata alignment errors between 4-channel images and 3-channel masks.

## 15. Label Dimension Mismatch (4D vs 3D)
- **Error:** `AssertionError: ground truth has different shape from input` (e.g., shape containing a `0` or extra dimension).
- **Cause:** When stacking custom multi-channel labels, if the input labels are `(1, H, W, D)`, a simple `torch.stack` creates a `(3, 1, H, W, D)` tensor. This adds an extra spatial dimension of size 1, which confuses spatial transforms like `RandSpatialCropd`.
- **Fix:** Squeezed the input label map to `(H, W, D)` before stacking, resulting in a correct `(3, H, W, D)` multi-channel tensor.
