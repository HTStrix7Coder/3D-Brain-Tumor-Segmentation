import os
import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = (
    "D:/AICode/Medical Lab/BraTS/data/validation/"
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/"
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
)

MODALITY_KEYS   = ["t1n", "t1c", "t2w", "t2f"]
MODALITY_LABELS = {"t1n": "T1 Native", "t1c": "T1 Contrast", "t2w": "T2 Weighted", "t2f": "T2 FLAIR"}
MODALITY_COLORS = {"t1n": "#1f77b4", "t1c": "#ff7f0e", "t2w": "#2ca02c", "t2f": "#d62728"}

# Segmentation overlay colours (RGBA)
SEG_PALETTE = {
    "WT": (0.00, 0.80, 0.40, 0.45),   # green  – whole tumour
    "TC": (1.00, 0.85, 0.00, 0.50),   # yellow – tumour core
    "ET": (1.00, 0.20, 0.20, 0.60),   # red    – enhancing tumour
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_mri(patient_id: str):
    """Load all four MRI modalities via MONAI and return a dict of 3-D arrays."""
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Orientationd, DivisiblePadd, Compose,
    )
    loader = Compose([
        LoadImaged(keys=MODALITY_KEYS),
        EnsureChannelFirstd(keys=MODALITY_KEYS),
        Orientationd(keys=MODALITY_KEYS, axcodes="RAS"),
        DivisiblePadd(keys=MODALITY_KEYS, k=8),
    ])
    paths = {
        k: os.path.join(DATA_ROOT, patient_id, f"{patient_id}-{k}.nii.gz")
        for k in MODALITY_KEYS
    }
    data = loader(paths)
    # Strip the channel dimension added by EnsureChannelFirstd → (H, W, D)
    return {k: data[k][0] for k in MODALITY_KEYS}


def _load_segmentations(patient_res_dir: str):
    """Load WT / TC / ET binary segmentation masks."""
    return {
        label: nib.load(os.path.join(patient_res_dir, f"pred_{label}.nii.gz")).get_fdata()
        for label in ("WT", "TC", "ET")
    }


def _best_slice(seg_wt: np.ndarray) -> int:
    """Return the axial slice index with the most tumour voxels."""
    return int(np.argmax(np.sum(seg_wt, axis=(0, 1))))


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _rgba_overlay(base_grey: np.ndarray, mask: np.ndarray, colour: tuple) -> np.ndarray:
    """
    Blend a binary mask as a translucent colour on top of a greyscale slice.
    Returns an (H, W, 4) RGBA array ready for imshow.
    """
    h, w = base_grey.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 0] = base_grey
    rgba[..., 1] = base_grey
    rgba[..., 2] = base_grey
    rgba[..., 3] = 1.0
    r, g, b, a = colour
    where = mask > 0
    rgba[where, 0] = rgba[where, 0] * (1 - a) + r * a
    rgba[where, 1] = rgba[where, 1] * (1 - a) + g * a
    rgba[where, 2] = rgba[where, 2] * (1 - a) + b * a
    return rgba


def _ax_off(ax, title: str, fontsize: int = 14):
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=6)


def _append_colorbar(ax, im):
    """Real colorbar strip appended right of ax via make_axes_locatable."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.06)
    cb  = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, colors="#333333", length=2)
    cb.outline.set_edgecolor("#cccccc")


def _append_dummy_strip(ax):
    """Invisible strip same size as _append_colorbar — keeps neighbour panel equal width."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.06)
    cax.set_visible(False)

# ──────────────────────────────────────────────────────────────────────────────
# Main report function
# ──────────────────────────────────────────────────────────────────────────────

def create_report(patient_id: str, prediction_dir: str = "results/predictions"):
    # ── 1. Paths ──────────────────────────────────────────────────────────────
    patient_res_dir = os.path.join(prediction_dir, patient_id)
    seg_path        = os.path.join(patient_res_dir, "pred_WT.nii.gz")

    # ── 2. Auto-infer if predictions are missing ──────────────────────────────
    if not os.path.exists(seg_path):
        print(f"[INFO] Predictions not found for {patient_id}. Running auto-inference …")
        from src.inference.predict_uncertainty import predict_with_uncertainty
        predict_with_uncertainty(patient_id)

    unc_path     = os.path.join(patient_res_dir, "uncertainty_map.nii.gz")
    weights_path = os.path.join(patient_res_dir, "modality_weights.txt")

    # ── 3. Load data ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading MRI modalities for {patient_id} …")
    mri      = _load_mri(patient_id)
    segs     = _load_segmentations(patient_res_dir)
    unc      = nib.load(unc_path).get_fdata()
    slice_z  = _best_slice(segs["WT"])

    print(f"[INFO] Using axial slice {slice_z} (max WT coverage).")

    # ── 4. Pre-extract 2-D slices ─────────────────────────────────────────────
    mri_slices = {k: np.rot90(_normalise(mri[k][:, :, slice_z])) for k in MODALITY_KEYS}
    seg_slices = {k: np.rot90(segs[k][:, :, slice_z])             for k in ("WT", "TC", "ET")}
    unc_slice  = np.rot90(unc[:, :, slice_z])

    # ── 5. Figure & GridSpec ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 13), facecolor="white")
    fig.suptitle(
        f"NeuroGate-VAE  ·  Clinical Analysis Report\n"
        f"Patient: {patient_id}   |   Axial Slice: {slice_z}",
        fontsize=22, fontweight="bold", color="#111111", y=0.98,
    )

    gs = GridSpec(
        2, 4, figure=fig,
        hspace=0.12, wspace=0.08,          # tight: minimal gaps
        top=0.91, bottom=0.04, left=0.03, right=0.97,
    )

    # ── 6. Row 0 – MRI modalities with WT contour overlay ────────────────────
    for col, key in enumerate(MODALITY_KEYS):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(mri_slices[key], cmap="gray", interpolation="bilinear")
        # Crisp WT contour in cyan – readable on grey brain
        ax.contour(seg_slices["WT"], levels=[0.5], colors=["#00aaff"], linewidths=0.9, alpha=0.8)
        _ax_off(ax, MODALITY_LABELS[key], fontsize=13)
        # Small colour badge bottom-right
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.73, 0.02), 0.25, 0.10,
            boxstyle="round,pad=0.01", transform=ax.transAxes,
            facecolor=MODALITY_COLORS[key], alpha=0.90, zorder=5,
        ))
        ax.text(0.855, 0.07, key.upper(), transform=ax.transAxes,
                color="white", fontsize=9, fontweight="bold",
                ha="center", va="center", zorder=6)

    # ── 7. Row 1, Col 0 – Composite segmentation overlay ─────────────────────
    ax_seg = fig.add_subplot(gs[1, 0])
    base = mri_slices["t1c"]                            # T1c is most informative
    h, w = base.shape

    # Build RGBA from scratch: start with greyscale, then blend each mask in order.
    # WT (green) goes first, TC (yellow) overwrites WT, ET (red) overwrites TC.
    # Each layer is: out = existing * (1-a) + colour * a
    rgba = np.stack([base, base, base, np.ones((h, w), dtype=np.float32)], axis=-1)
    for label in ("WT", "TC", "ET"):
        r, g, b, a = SEG_PALETTE[label]
        where = seg_slices[label] > 0
        rgba[where, 0] = rgba[where, 0] * (1 - a) + r * a
        rgba[where, 1] = rgba[where, 1] * (1 - a) + g * a
        rgba[where, 2] = rgba[where, 2] * (1 - a) + b * a

    ax_seg.imshow(rgba, interpolation="bilinear")
    _ax_off(ax_seg, "Segmentation Overlay (T1c)", fontsize=13)

    legend_patches = [
        mpatches.Patch(facecolor=SEG_PALETTE["WT"][:3], alpha=0.85, label="Whole Tumour (WT)"),
        mpatches.Patch(facecolor=SEG_PALETTE["TC"][:3], alpha=0.85, label="Tumour Core (TC)"),
        mpatches.Patch(facecolor=SEG_PALETTE["ET"][:3], alpha=0.85, label="Enhancing Tumour (ET)"),
    ]
    ax_seg.legend(
        handles=legend_patches, loc="lower left",
        fontsize=8, framealpha=0.75,
        labelcolor="#111111", facecolor="white", edgecolor="#cccccc",
    )
    _append_dummy_strip(ax_seg)   # match width of uncertainty panel

    # ── 8. Row 1, Col 1 – Uncertainty map ────────────────────────────────────
    ax_unc = fig.add_subplot(gs[1, 1])
    im_unc = ax_unc.imshow(unc_slice, cmap="hot", interpolation="bilinear")
    _ax_off(ax_unc, "VAE Uncertainty Map", fontsize=13)
    _append_colorbar(ax_unc, im_unc)

    # ── 9. Row 1, Col 2-3 – Modality gating importance ───────────────────────
    ax_gate = fig.add_subplot(gs[1, 2:])
    ax_gate.set_facecolor("white")
    # Shift right so y-axis labels don't overlap the uncertainty map panel.
    _pos = ax_gate.get_position()
    ax_gate.set_position([_pos.x0 + 0.04, _pos.y0, _pos.width - 0.04, _pos.height])

    if os.path.exists(weights_path):
        names, vals = [], []
        with open(weights_path) as f:
            for line in f:
                line = line.strip()
                if ": " in line:
                    n, v = line.split(": ", 1)
                    names.append(n.strip())
                    vals.append(float(v.strip()))

        bar_colours = [MODALITY_COLORS.get(n.lower(), "#888888") for n in names]
        bars = ax_gate.barh(names, vals, color=bar_colours, alpha=0.88, height=0.55)

        ax_gate.set_xlim(0, max(vals) * 1.35 if vals else 1.4)
        ax_gate.set_title("Modality Gating Importance", fontsize=15, fontweight="bold",
                           pad=12, color="#111111")
        ax_gate.invert_yaxis()
        ax_gate.tick_params(axis="y", labelsize=11, colors="#222222", pad=8)
        ax_gate.tick_params(axis="x", labelsize=9,  colors="#555555")
        ax_gate.set_xlabel("Attention Weight", fontsize=10, color="#555555", labelpad=6)
        for spine in ("top", "right"):
            ax_gate.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax_gate.spines[spine].set_color("#cccccc")

        for bar, val in zip(bars, vals):
            ax_gate.text(
                bar.get_width() + max(vals) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center", ha="left",
                fontweight="bold", fontsize=11, color="#111111",
            )
    else:
        ax_gate.text(0.5, 0.5, "modality_weights.txt not found",
                     ha="center", va="center", color="#888888", fontsize=12)
        ax_gate.axis("off")

    # ── 10. Apply white style to all axes ────────────────────────────────────
    for ax in fig.axes:
        if ax.get_images() or ax is ax_seg:   # image panels – keep black bg
            ax.set_facecolor("black")
        else:
            ax.set_facecolor("white")
        ax.title.set_color("#111111")

    # ── 11. Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(patient_res_dir, f"{patient_id}_report.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] Report saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "BraTS-GLI-00001-000"
    create_report(pid)