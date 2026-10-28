"""
Image preprocessing and conservative augmentation pipeline for knee X-rays.
Preserves anatomical aspect ratios via letterbox padding, standardizes normalization,
and avoids unrealistic distortions.
"""
import os
from typing import Tuple
from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def resize_and_pad_radiograph(img: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Resize image preserving anatomical aspect ratio and pad with black background.
    Prevents bone distortion.
    """
    # Ensure image is in RGB mode for pretrained backbones
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    ratio = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Pad to square target_size
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top
    padded = ImageOps.expand(resized, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
    return padded

class KneeXRayTrainTransform:
    """
    Conservative training augmentation for knee radiographs.
    Applies small rotations, slight scaling/translation, and mild contrast/brightness.
    Never uses vertical flips or anatomically unrealistic shears.
    """
    def __init__(self, target_size: int = 224, allow_horizontal_flip: bool = False):
        self.target_size = target_size
        self.allow_horizontal_flip = allow_horizontal_flip
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. Letterbox pad to square
        img = resize_and_pad_radiograph(img, self.target_size)

        # 2. Conservative random rotation (-10 to +10 degrees)
        angle = float(np.random.uniform(-10.0, 10.0))
        img = TF.rotate(img, angle, fill=0)

        # 3. Slight random affine translation (-5% to +5%) and scale (0.95 to 1.05)
        scale = float(np.random.uniform(0.95, 1.05))
        max_trans = int(self.target_size * 0.05)
        translate = [int(np.random.uniform(-max_trans, max_trans)), int(np.random.uniform(-max_trans, max_trans))]
        img = TF.affine(img, angle=0, translate=translate, scale=scale, shear=0, fill=0)

        # 4. Mild brightness & contrast (0.9 to 1.1)
        bright = float(np.random.uniform(0.90, 1.10))
        contrast = float(np.random.uniform(0.90, 1.10))
        img = TF.adjust_brightness(img, bright)
        img = TF.adjust_contrast(img, contrast)

        # 5. Optional horizontal flip (disabled by default for laterality preservation)
        if self.allow_horizontal_flip and np.random.rand() > 0.5:
            img = TF.hflip(img)

        # 6. Tensor conversion and ImageNet normalization
        tensor = TF.to_tensor(img)
        tensor = self.normalize(tensor)
        return tensor

class KneeXRayEvalTransform:
    """
    Deterministic evaluation / test preprocessing.
    Letterbox pad, convert to tensor, and apply ImageNet normalization.
    """
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        padded = resize_and_pad_radiograph(img, self.target_size)
        tensor = TF.to_tensor(padded)
        tensor = self.normalize(tensor)
        return tensor

def generate_preprocessing_qc_figure(
    sample_image_path: str,
    output_path: str = "outputs/figures/preprocessing_qc.png",
    target_size: int = 224,
) -> None:
    """Generate a visual quality control figure showing original vs letterboxed vs augmented copies."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with Image.open(sample_image_path) as orig:
        orig_rgb = orig.convert("RGB")
        eval_t = KneeXRayEvalTransform(target_size)
        train_t = KneeXRayTrainTransform(target_size)

        padded = resize_and_pad_radiograph(orig_rgb, target_size)
        aug1 = train_t(orig_rgb)
        aug2 = train_t(orig_rgb)
        aug3 = train_t(orig_rgb)

    # Denormalize for plotting
    def denorm(t: torch.Tensor) -> np.ndarray:
        arr = t.clone().detach().cpu().numpy()
        for i in range(3):
            arr[i] = arr[i] * IMAGENET_STD[i] + IMAGENET_MEAN[i]
        arr = np.clip(arr, 0.0, 1.0)
        return np.transpose(arr, (1, 2, 0))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(orig_rgb)
    axes[0].set_title(f"Original X-Ray\n{orig_rgb.size[0]}x{orig_rgb.size[1]}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(padded)
    axes[1].set_title(f"Letterbox Padded\n{target_size}x{target_size}", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(denorm(aug1))
    axes[2].set_title("Conservative Aug 1\n(Rotation & Scale)", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(denorm(aug2))
    axes[3].set_title("Conservative Aug 2\n(Translation & Brightness)", fontsize=10)
    axes[3].axis("off")

    axes[4].imshow(denorm(aug3))
    axes[4].set_title("Conservative Aug 3\n(Contrast & Shift)", fontsize=10)
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
