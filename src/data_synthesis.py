"""
Data synthesis and class-conditioned generative latent expansion module.
Implements manifold-consistent convex interpolation and stochastic latent perturbation
strictly on training embeddings to remediate class imbalances without test-set leakage.
Includes visual quality control generation for synthetic sample inspection.
"""
import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

class LatentManifoldSynthesizer:
    """
    Class-conditioned latent manifold synthesizer.
    Generates high-fidelity synthetic feature vectors for underrepresented classes
    strictly within the training partition manifold.
    """
    def __init__(self, target_per_class: Optional[int] = None, noise_std: float = 0.01, random_seed: int = 42):
        self.target_per_class = target_per_class
        self.noise_std = noise_std
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def fit_resample(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize latent features strictly from training instances.
        Never accesses validation or test samples.
        """
        classes, counts = np.unique(y_train, return_counts=True)
        max_count = self.target_per_class or max(counts)

        X_syn_list = [X_train]
        y_syn_list = [y_train]

        for c, count in zip(classes, counts):
            if count < max_count:
                needed = max_count - count
                X_c = X_train[y_train == c]
                n_c = len(X_c)

                # Sample pairs from the same class manifold
                idx1 = self.rng.choice(n_c, size=needed)
                idx2 = self.rng.choice(n_c, size=needed)

                # Convex interpolation with stochastic Dirichlet/uniform weights
                lam = self.rng.uniform(0.2, 0.8, size=(needed, 1))
                syn_feats = lam * X_c[idx1] + (1.0 - lam) * X_c[idx2]

                # Subtle manifold gaussian jitter to prevent exact colinearity
                syn_feats += self.rng.normal(0.0, self.noise_std, size=syn_feats.shape)

                X_syn_list.append(syn_feats)
                y_syn_list.append(np.full(needed, c))

        X_resampled = np.vstack(X_syn_list)
        y_resampled = np.concatenate(y_syn_list)

        # Shuffle training set
        shuffle_idx = self.rng.permutation(len(y_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


def generate_synthetic_samples_qc(
    train_df: pd.DataFrame,
    output_path: str = "outputs/figures/synthetic_data_qc.png",
    num_pairs: int = 3,
) -> None:
    """
    Generate visual demonstration of intra-class medical radiograph blending (MixUp)
    and save a side-by-side diagnostic QC figure.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classes = ["Normal", "Osteopenia", "Osteoporosis"]

    fig, axes = plt.subplots(len(classes), 3, figsize=(10, 3 * len(classes)))
    plt.suptitle("AI Data Generation & Synthesis QC: Intra-Class Bone Manifold Blending", fontsize=13, y=0.99)

    for i, cls in enumerate(classes):
        cls_df = train_df[train_df["class_label"] == cls]
        if len(cls_df) < 2:
            continue
        p1 = cls_df.iloc[0]["image_path"]
        p2 = cls_df.iloc[1]["image_path"]

        img1 = Image.open(p1).convert("L").resize((224, 224))
        img2 = Image.open(p2).convert("L").resize((224, 224))

        arr1 = np.array(img1, dtype=np.float32) / 255.0
        arr2 = np.array(img2, dtype=np.float32) / 255.0

        # Convex blend (lambda = 0.5)
        blended = 0.5 * arr1 + 0.5 * arr2

        axes[i, 0].imshow(arr1, cmap="gray")
        axes[i, 0].set_title(f"Source Radiograph A ({cls})", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(blended, cmap="gray")
        axes[i, 1].set_title(f"Synthesized Radiograph (λ=0.5 {cls})", fontsize=9, color="darkblue")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(arr2, cmap="gray")
        axes[i, 2].set_title(f"Source Radiograph B ({cls})", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
