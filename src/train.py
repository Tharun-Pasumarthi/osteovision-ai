"""
PyTorch training pipeline for image-only CNNs and multimodal models.
Implements AdamW, cosine annealing, square-root smoothed class weighting,
early stopping on validation macro-F1, temperature scaling probability calibration,
checkpoint management, and Windows-safe DataLoaders.
"""
import os
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.image_transforms import KneeXRayTrainTransform, KneeXRayEvalTransform
from src.evaluate import compute_comprehensive_metrics

CLASS_TO_IDX = {"Normal": 0, "Osteopenia": 1, "Osteoporosis": 2}
CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]

class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to calibrate deep neural network logits.
    Fitted exclusively on validation partition logits.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def _step():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(_step)
        return float(self.temperature.item())

class KneeXRayDataset(Dataset):
    """PyTorch Dataset for image-only classification."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = KneeXRayEvalTransform()(img)
        label = CLASS_TO_IDX[row["class_label"]]
        return tensor, label, row["patient_id"]

class MultimodalKneeDataset(Dataset):
    """PyTorch Dataset for image-plus-clinical multimodal classification."""
    def __init__(self, df: pd.DataFrame, clinical_features: np.ndarray, transform=None):
        self.df = df.reset_index(drop=True)
        self.clinical_features = torch.tensor(clinical_features, dtype=torch.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img_tensor = self.transform(img)
            else:
                img_tensor = KneeXRayEvalTransform()(img)
        clin_tensor = self.clinical_features[idx]
        label = CLASS_TO_IDX[row["class_label"]]
        return img_tensor, clin_tensor, label, row["patient_id"]

def compute_class_weights(
    train_labels: List[int],
    num_classes: int = 3,
    device: torch.device = torch.device("cpu"),
    smoothing: str = "sqrt",
) -> torch.Tensor:
    """
    Compute class weights from training partition only.
    Uses square-root smoothing by default to prevent aggressive over-weighting
    of minority classes that triggers severe Normal false positives.
    """
    counts = np.bincount(train_labels, minlength=num_classes)
    total = len(train_labels)

    if smoothing == "sqrt":
        # Square-root smoothed inverse weighting
        raw_weights = np.sqrt(total / (num_classes * np.maximum(counts, 1).astype(float)))
        # Normalize weights so mean is 1.0
        weights = raw_weights / np.mean(raw_weights)
    elif smoothing == "none":
        weights = np.ones(num_classes, dtype=float)
    else:
        # Full inverse frequency
        weights = total / (num_classes * np.maximum(counts, 1).astype(float))

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights_tensor

def train_image_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.0003,
    weight_decay: float = 0.01,
    patience: int = 5,
    device: Optional[torch.device] = None,
    checkpoint_path: str = "outputs/checkpoints/best_image_model.pt",
    class_weights: Optional[torch.Tensor] = None,
    logger=None,
) -> Dict[str, Any]:
    """Train image-only CNN baseline with early stopping on validation macro-F1."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_val_logits: Optional[torch.Tensor] = None
    best_val_targets: Optional[torch.Tensor] = None
    fitted_temp = 1.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_acc": [],
        "val_balanced_acc": [],
        "val_normal_recall": [],
        "val_normal_fpr": [],
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs, raw_logits_list = [], [], [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                raw_logits_list.append(outputs.cpu())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_metrics = compute_comprehensive_metrics(
            np.array(val_targets), np.array(val_preds), np.array(val_probs)
        )
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_bal_acc = val_metrics["balanced_accuracy"]
        norm_rec = val_metrics["per_class"]["Normal"]["recall"]

        cm = val_metrics["confusion_matrix"]
        norm_support = cm[0][0] + cm[0][1] + cm[0][2]
        norm_fpr = (cm[0][1] + cm[0][2]) / max(norm_support, 1)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_bal_acc)
        history["val_normal_recall"].append(norm_rec)
        history["val_normal_fpr"].append(norm_fpr)

        if logger:
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | "
                f"Normal Recall: {norm_rec:.4f} | Normal FPR: {norm_fpr:.4f}"
            )

        # Early stopping and checkpointing on validation macro-F1
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_val_logits = torch.cat(raw_logits_list, dim=0)
            best_val_targets = torch.tensor(val_targets, dtype=torch.long)

            # Fit temperature scaling calibration on validation logits
            try:
                scaler = TemperatureScaler()
                fitted_temp = scaler.fit(best_val_logits, best_val_targets)
            except Exception:
                fitted_temp = 1.0

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_metrics": val_metrics,
                "temperature": fitted_temp,
                "history": history,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if logger:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    training_duration = time.time() - start_time
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "training_duration_seconds": round(training_duration, 2),
        "temperature": fitted_temp,
        "history": history,
    }

def train_multimodal_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.0003,
    weight_decay: float = 0.01,
    patience: int = 5,
    device: Optional[torch.device] = None,
    checkpoint_path: str = "outputs/checkpoints/best_multimodal_model.pt",
    class_weights: Optional[torch.Tensor] = None,
    logger=None,
) -> Dict[str, Any]:
    """Train image+clinical multimodal model with early stopping on validation macro-F1."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_val_logits: Optional[torch.Tensor] = None
    best_val_targets: Optional[torch.Tensor] = None
    fitted_temp = 1.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_acc": [],
        "val_balanced_acc": [],
        "val_normal_recall": [],
        "val_normal_fpr": [],
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, clinical, labels, _ in train_loader:
            images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs, raw_logits_list = [], [], [], []
        with torch.no_grad():
            for images, clinical, labels, _ in val_loader:
                images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)
                outputs = model(images, clinical)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                raw_logits_list.append(outputs.cpu())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_metrics = compute_comprehensive_metrics(
            np.array(val_targets), np.array(val_preds), np.array(val_probs)
        )
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_bal_acc = val_metrics["balanced_accuracy"]
        norm_rec = val_metrics["per_class"]["Normal"]["recall"]

        cm = val_metrics["confusion_matrix"]
        norm_support = cm[0][0] + cm[0][1] + cm[0][2]
        norm_fpr = (cm[0][1] + cm[0][2]) / max(norm_support, 1)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_bal_acc)
        history["val_normal_recall"].append(norm_rec)
        history["val_normal_fpr"].append(norm_fpr)

        if logger:
            logger.info(
                f"[Multimodal] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | "
                f"Normal Recall: {norm_rec:.4f} | Normal FPR: {norm_fpr:.4f}"
            )

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_val_logits = torch.cat(raw_logits_list, dim=0)
            best_val_targets = torch.tensor(val_targets, dtype=torch.long)

            try:
                scaler = TemperatureScaler()
                fitted_temp = scaler.fit(best_val_logits, best_val_targets)
            except Exception:
                fitted_temp = 1.0

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_metrics": val_metrics,
                "temperature": fitted_temp,
                "history": history,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if logger:
                    logger.info(f"[Multimodal] Early stopping triggered at epoch {epoch}.")
                break

    training_duration = time.time() - start_time
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "training_duration_seconds": round(training_duration, 2),
        "temperature": fitted_temp,
        "history": history,
    }
