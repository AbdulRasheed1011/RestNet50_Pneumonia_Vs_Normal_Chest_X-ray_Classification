import os
import random
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def list_images(folder: str | Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    folder = Path(folder)
    return [str(p) for p in folder.rglob("*") if p.suffix.lower() in exts]

def class_map_from_train_dir(train_dir: str | Path):
    """Returns (class_to_idx, idx_to_class). Classes are subfolder names sorted alphabetically."""
    train_dir = Path(train_dir)
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class

def save_json(obj, path: str | Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)

def preprocess_resnet50(x_uint8: np.ndarray) -> np.ndarray:
    """x_uint8: [N, H, W, 3] uint8 RGB -> float32 preprocessed for ResNet50."""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    x = x_uint8.astype(np.float32)
    return preprocess_input(x)

def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Binary/multi-class class weights from labels y (ints)."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}