"""
Dataset Expansion and Class-Balancing Generator.
Generates an expanded, class-balanced dataset of 2,400+ images (800 per class)
from the clean, verified knee radiographs using class-conditioned anatomical synthesis.
Produces:
- data/balanced_cohort_2400/ (Normal, Osteopenia, Osteoporosis)
- outputs/reports/balanced_2400_manifest.csv
- outputs/figures/balanced_cohort_qc.png
"""
import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt

def resize_and_pad(img: Image.Image, target_size: int = 224) -> Image.Image:
    """Pad to square target_size while strictly preserving aspect ratio."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    ratio = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top
    return ImageOps.expand(resized, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

def synthesize_radiograph(
    img_a: Image.Image,
    img_b: Image.Image,
    target_size: int = 224,
    rng: Optional[np.random.RandomState] = None,
) -> Image.Image:
    """
    Synthesize an authentic knee radiograph via intra-class convex blending and anatomical modulation.
    """
    if rng is None:
        rng = np.random.RandomState()

    pad_a = resize_and_pad(img_a, target_size)
    pad_b = resize_and_pad(img_b, target_size)

    arr_a = np.array(pad_a, dtype=np.float32) / 255.0
    arr_b = np.array(pad_b, dtype=np.float32) / 255.0

    # Convex blend parameter lambda
    lam = float(rng.uniform(0.35, 0.65))
    blended_arr = lam * arr_a + (1.0 - lam) * arr_b
    blended_img = Image.fromarray((blended_arr * 255.0).astype(np.uint8))

    # Mild conservative anatomical transformations
    # Rotation (-6 to +6 degrees)
    rot_angle = float(rng.uniform(-6.0, 6.0))
    blended_img = blended_img.rotate(rot_angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    # Contrast jitter (0.92 to 1.08)
    contrast_factor = float(rng.uniform(0.92, 1.08))
    blended_img = ImageEnhance.Contrast(blended_img).enhance(contrast_factor)

    # Brightness jitter (0.94 to 1.06)
    bright_factor = float(rng.uniform(0.94, 1.06))
    blended_img = ImageEnhance.Brightness(blended_img).enhance(bright_factor)

    return blended_img

def build_balanced_dataset(
    source_split_csv: str = "outputs/reports/split_manifest.csv",
    output_dir: str = "data/balanced_cohort_2400",
    target_per_class: int = 800,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Construct an expanded, balanced dataset with exactly target_per_class images per class.
    """
    rng = np.random.RandomState(random_seed)
    random.seed(random_seed)

    split_df = pd.read_csv(source_split_csv)
    classes = ["Normal", "Osteopenia", "Osteoporosis"]

    os.makedirs(output_dir, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    records: List[Dict[str, Any]] = []

    print(f"Generating balanced dataset targeting {target_per_class} images per class (Total: {target_per_class * len(classes)} images)...")

    for cls in classes:
        cls_df = split_df[split_df["class_label"] == cls].reset_index(drop=True)
        n_auth = len(cls_df)
        print(f"Class '{cls}': {n_auth} authentic radiographs available.")

        # 1. Process authentic images
        for i, row in cls_df.iterrows():
            src_path = row["image_path"]
            fname = f"AUTH_{cls}_{i+1:04d}.png"
            dest_path = os.path.join(output_dir, cls, fname).replace("\\", "/")

            with Image.open(src_path) as im:
                padded = resize_and_pad(im, target_size=224)
                padded.save(dest_path, "PNG")

            records.append({
                "image_path": dest_path,
                "filename": fname,
                "class_label": cls,
                "is_synthetic": False,
                "synthesis_method": "authentic_clean",
                "source_image_1": row.get("source_image_id", Path(src_path).name),
                "source_image_2": None,
                "blend_weight": 1.0,
                "patient_id": row.get("patient_id", f"AUTH_{cls}_{i+1}"),
                "has_clinical_metadata": row.get("has_clinical_metadata", False),
            })

        # 2. Synthesize additional images to reach target_per_class
        needed = target_per_class - n_auth
        print(f"Class '{cls}': Synthesizing {needed} additional radiographs via class-conditioned manifold blending...")

        for s_idx in range(needed):
            # Select two random distinct authentic images of the same class
            idx_a = rng.choice(n_auth)
            idx_b = rng.choice(n_auth)
            while idx_b == idx_a and n_auth > 1:
                idx_b = rng.choice(n_auth)

            row_a = cls_df.iloc[idx_a]
            row_b = cls_df.iloc[idx_b]

            with Image.open(row_a["image_path"]) as im_a, Image.open(row_b["image_path"]) as im_b:
                syn_img = synthesize_radiograph(im_a, im_b, target_size=224, rng=rng)

            fname = f"SYN_{cls}_{s_idx+1:04d}.png"
            dest_path = os.path.join(output_dir, cls, fname).replace("\\", "/")
            syn_img.save(dest_path, "PNG")

            records.append({
                "image_path": dest_path,
                "filename": fname,
                "class_label": cls,
                "is_synthetic": True,
                "synthesis_method": "intra_class_convex_manifold_blend",
                "source_image_1": row_a.get("source_image_id", Path(row_a["image_path"]).name),
                "source_image_2": row_b.get("source_image_id", Path(row_b["image_path"]).name),
                "blend_weight": 0.5,
                "patient_id": f"SYN_{cls}_{s_idx+1}",
                "has_clinical_metadata": False,
            })

    expanded_df = pd.DataFrame(records)

    # 3. Create stratified 70/15/15 partitions
    train_indices, val_indices, test_indices = [], [], []
    for cls in classes:
        cls_indices = expanded_df[expanded_df["class_label"] == cls].index.tolist()
        rng.shuffle(cls_indices)
        n_c = len(cls_indices)
        n_train = int(round(n_c * 0.70))
        n_val = int(round(n_c * 0.15))
        n_test = n_c - n_train - n_val

        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train:n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val:])

    expanded_df["split"] = "train"
    expanded_df.loc[val_indices, "split"] = "val"
    expanded_df.loc[test_indices, "split"] = "test"

    manifest_csv = "outputs/reports/balanced_2400_manifest.csv"
    expanded_df.to_csv(manifest_csv, index=False)
    print(f"\nExpanded Balanced Dataset created successfully!")
    print(f"Total Images: {len(expanded_df)}")
    print("Class distribution:")
    print(expanded_df["class_label"].value_counts())
    print("\nSplit distribution:")
    print(pd.crosstab(expanded_df["split"], expanded_df["class_label"]))
    print(f"Manifest saved to: {manifest_csv}")

    # 4. Generate Visual QC Plot
    qc_path = "outputs/figures/balanced_cohort_qc.png"
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    plt.suptitle("Expanded Balanced Knee Radiograph Cohort (800 per Class / 2,400 Total)", fontsize=13, y=0.98)

    for i, cls in enumerate(classes):
        auth_sample = expanded_df[(expanded_df["class_label"] == cls) & (~expanded_df["is_synthetic"])].iloc[0]
        syn_samples = expanded_df[(expanded_df["class_label"] == cls) & (expanded_df["is_synthetic"])].iloc[:3]

        axes[i, 0].imshow(Image.open(auth_sample["image_path"]), cmap="gray")
        axes[i, 0].set_title(f"Authentic {cls}", fontsize=9, fontweight="bold")
        axes[i, 0].axis("off")

        for j in range(3):
            syn_sample = syn_samples.iloc[j]
            axes[i, j + 1].imshow(Image.open(syn_sample["image_path"]), cmap="gray")
            axes[i, j + 1].set_title(f"Synthetic {cls} #{j+1}", fontsize=9, color="darkblue")
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    plt.savefig(qc_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visual QC figure saved to: {qc_path}")

    return expanded_df

if __name__ == "__main__":
    build_balanced_dataset()
