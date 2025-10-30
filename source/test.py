"""
Unified inference & evaluation tool.

Modes:
1) Batch inference on unlabeled folder (default)
   python source/test.py --test_dir data/test --model_path outputs/resnet50_best.keras --out_csv outputs/test_predictions.csv

2) Single-image prediction
   python source/test.py --image_path path/to/image.jpg --model_path outputs/resnet50_best.keras

3) Evaluation (NPZ from convert script)
   python source/test.py --eval --use_val_npz --data_npz_dir data/processed --model_path outputs/resnet50_best.keras

4) Evaluation (labeled folder with class subdirs)
   python source/test.py --eval --val_dir data/val --model_path outputs/resnet50_best.keras

Quality-of-life:
- If --model_path is omitted, the script will try to auto-pick the newest *.keras in outputs/
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from tqdm import tqdm

# --------- helpers shared with older scripts ----------
def load_json(path: str | Path):
    import json
    with open(path) as f:
        return json.load(f)

def preprocess_resnet50(x_uint8: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.resnet50 import preprocess_input
    x = x_uint8.astype(np.float32)
    return preprocess_input(x)

def auto_find_model(default_dir="outputs"):
    """Pick the newest *.keras model if user didn't pass --model_path."""
    out = Path(default_dir)
    if not out.exists():
        return None
    candidates = sorted(out.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None

def read_and_resize(path: Path, img_size: int):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img

def load_labeled_folder(val_dir: Path, img_size: int, class_to_idx: dict[str, int]):
    X_list, y_list = [], []
    for cls, idx in class_to_idx.items():
        for p in sorted((val_dir / cls).rglob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                continue
            img = read_and_resize(p, img_size)
            if img is None:
                continue
            X_list.append(img)
            y_list.append(idx)
    if not X_list:
        raise RuntimeError(f"No images found under labeled dir: {val_dir}")
    X = np.array(X_list, dtype=np.uint8)
    y = np.array(y_list, dtype=np.int32)
    return X, y

def ensure_out_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ---------------- main logic -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    # Common
    ap.add_argument("--model_path", type=str, help="Path to trained model (.keras). If omitted, picks newest in outputs/")
    ap.add_argument("--meta_path", type=str, default="data/processed/meta.json", help="Path to meta.json")
    ap.add_argument("--data_npz_dir", type=str, default="data/processed", help="Folder with train.npz/val.npz/meta.json")

    # Batch inference
    ap.add_argument("--test_dir", type=str, default="data/test", help="Unlabeled test images folder")
    ap.add_argument("--out_csv", type=str, default="outputs/test_predictions.csv")

    # Single image
    ap.add_argument("--image_path", type=str, help="Single image path for quick prediction")

    # Evaluation
    ap.add_argument("--eval", action="store_true", help="Run evaluation instead of inference")
    ap.add_argument("--use_val_npz", action="store_true", help="Evaluate on data/processed/val.npz")
    ap.add_argument("--val_dir", type=str, help="Evaluate on labeled folder with class subdirs")

    return ap.parse_args()

def load_model_or_die(model_path: str | None):
    if not model_path:
        model_path = auto_find_model("outputs")
        if model_path:
            print(f"[info] --model_path not provided, using newest model: {model_path}")
    if not model_path:
        print("error: --model_path is required (no model found in outputs/).", file=sys.stderr)
        sys.exit(2)
    if not Path(model_path).exists():
        print(f"error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(2)
    return tf.keras.models.load_model(model_path), model_path

def main():
    args = parse_args()

    # Load metadata (size + classes)
    meta = load_json(args.meta_path)
    img_size = int(meta["img_size"])
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    class_to_idx = {v: int(k) for k, v in meta["idx_to_class"].items()}

    model, used_model_path = load_model_or_die(args.model_path)

    # --------- Single image mode ---------
    if args.image_path:
        p = Path(args.image_path)
        if not p.exists():
            print(f"error: image not found: {p}", file=sys.stderr)
            sys.exit(2)
        img = read_and_resize(p, img_size)
        x = np.expand_dims(img, 0).astype(np.uint8)
        xp = preprocess_resnet50(x)
        pred = model.predict(xp, verbose=0)

        if pred.ndim == 2 and pred.shape[1] == 1:
            prob_pos = float(pred.squeeze())
            pred_idx = int(prob_pos >= 0.5)
            pred_class = idx_to_class[pred_idx]
            print(f"[model: {used_model_path}] {p.name} -> {pred_class}  (prob_PNEUMONIA={prob_pos:.4f})")
        else:
            idx = int(np.argmax(pred, axis=1)[0])
            prob = float(np.max(pred))
            print(f"[model: {used_model_path}] {p.name} -> {idx_to_class[idx]}  (prob={prob:.4f})")
        return

    # --------- Evaluation mode ---------
    if args.eval:
        if args.val_dir:
            X, y = load_labeled_folder(Path(args.val_dir), img_size, class_to_idx)
        else:
            # default to val.npz if --use_val_npz or no val_dir is given
            if not args.use_val_npz:
                print("[info] --eval set without --val_dir; falling back to --use_val_npz", file=sys.stderr)
            val_npz = np.load(Path(args.data_npz_dir) / "val.npz")
            X, y = val_npz["images"], val_npz["labels"]

        Xp = preprocess_resnet50(X)
        preds = model.predict(Xp, verbose=0)

        if preds.ndim == 2 and preds.shape[1] == 1:
            probs = preds.squeeze()
            yhat = (probs >= 0.5).astype(int)
        else:
            yhat = preds.argmax(1)

        acc = float((yhat == y).mean())
        print(f"Accuracy: {acc:.4f}")

        # optional sklearn report
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            print(classification_report(y, yhat, target_names=[idx_to_class[i] for i in sorted(idx_to_class)]))
            print("Confusion matrix:\n", confusion_matrix(y, yhat))
        except Exception:
            pass
        return

    # --------- Batch inference mode (default) ---------
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"error: test_dir not found: {test_dir}", file=sys.stderr)
        sys.exit(2)

    test_paths = [p for p in sorted(test_dir.iterdir()) if p.is_file()]
    if not test_paths:
        print(f"error: no files in test_dir: {test_dir}", file=sys.stderr)
        sys.exit(2)

    X_list, names = [], []
    for p in tqdm(test_paths, desc="Reading test images"):
        img = read_and_resize(p, img_size)
        if img is None:
            continue
        X_list.append(img)
        names.append(p.name)

    X = np.array(X_list, dtype=np.uint8)
    Xp = preprocess_resnet50(X)
    preds = model.predict(Xp, verbose=0)

    out_csv = Path(args.out_csv)
    ensure_out_parent(out_csv)

    if preds.ndim == 2 and preds.shape[1] == 1:
        prob_pos = preds.squeeze()
        pred_idx = (prob_pos >= 0.5).astype(int)
        pred_class = [idx_to_class[int(i)] for i in pred_idx]
        df = pd.DataFrame({"filename": names, "prob_PNEUMONIA": prob_pos, "pred_class": pred_class})
    else:
        pred_idx = preds.argmax(1)
        pred_class = [idx_to_class[int(i)] for i in pred_idx]
        max_prob = preds.max(1)
        df = pd.DataFrame({"filename": names, "pred_class": pred_class, "prob_max": max_prob})

    df.to_csv(out_csv, index=False)
    print(f"Saved predictions -> {out_csv}")

if __name__ == "__main__":
    main()