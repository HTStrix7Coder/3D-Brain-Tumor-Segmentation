"""
BraTS Evaluation Metrics
========================
Dice Score and Hausdorff Distance for WT, TC, ET regions.
"""

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose


def get_metrics():
    """
    Returns metric objects for BraTS evaluation.
    Output channels: [WT, TC, ET]
    """
    dice_metric = DiceMetric(
        include_background=True,   # All 3 channels are tumor regions
        reduction="mean_batch",    
    )

    hausdorff_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,             
        reduction="mean_batch",
    )

    return dice_metric, hausdorff_metric


def get_post_transforms():
    """
    Post-processing for model output:
    - Sigmoid activation (multi-label, not softmax)
    - Threshold at 0.5 to get binary masks
    - Convert to float32 (fixes linalg.vector_norm error)
    """
    post_sigmoid = Activations(sigmoid=True)
    post_pred = Compose([
        AsDiscrete(threshold=0.5),
        lambda x: x.float()
    ])
    return post_sigmoid, post_pred
