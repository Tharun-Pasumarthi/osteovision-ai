"""
Utility functions for configuration, reproducibility, logging, and environment detection.
"""
import os
import random
import logging
import json
import yaml
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Detect and return the appropriate PyTorch device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(name: str = "pipeline", log_file: str = None) -> logging.Logger:
    """Configure a standard logger for console and optional file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

def save_json(data: dict, file_path: str) -> None:
    """Safely save dictionary to formatted JSON."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def load_json(file_path: str) -> dict:
    """Load JSON file into a Python dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
