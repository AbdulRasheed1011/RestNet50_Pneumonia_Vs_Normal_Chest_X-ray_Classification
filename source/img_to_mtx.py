"""
Reads images with OpenCV, converts to matrices, makes a train/val split, and saves
compressed NPZ files plus metadata.

Usage:
  python source/convert_images_into_matrices.py \
      --train_dir data/train \
      --out_dir data/processed \
      --img_size 224 \
      --val_split 0.15
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from utils import set_seed, ensure_dir, class_map_from_train_dir, save_json

def load_class_images(root: Path, class_name: str, img_size: int):
    paths = sorted([p for p in (root / class_name).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")])
    X_list, y_list = [], []
    for p in tqdm(paths, desc=f"Loading {class_name}", leave=False):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue  # skip unreadable image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        X_list.append(img)         # keep uint8; preprocess later
        y_list.append(class_name)
    return X_list, y_list, [str(p) for p in paths]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    class_to_idx, idx_to_class = class_map_from_train_dir(train_dir)
    all_X, all_y, all_paths = [], [], []

    for cls in class_to_idx.keys():
        Xc, yc, pc = load_class_images(train_dir, cls, args.img_size)
        all_X.extend(Xc); all_y.extend([class_to_idx[c] for c in yc]); all_paths.extend(pc)

    # Shuffle
    idx = np.arange(len(all_X))
    np.random.shuffle(idx)
    X = np.array([all_X[i] for i in idx], dtype=np.uint8)
    y = np.array([all_y[i] for i in idx], dtype=np.int32)
    paths = np.array([all_paths[i] for i in idx])

    # Split
    n = len(X)
    n_val = int(args.val_split * n)
    X_val, y_val, paths_val = X[:n_val], y[:n_val], paths[:n_val]
    X_train, y_train, paths_train = X[n_val:], y[n_val:], paths[n_val:]

    # Save
    np.savez_compressed(out_dir / "train.npz", images=X_train, labels=y_train, paths=paths_train)
    np.savez_compressed(out_dir / "val.npz", images=X_val, labels=y_val, paths=paths_val)

    meta = {
        "img_size": args.img_size,
        "val_split": args.val_split,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_train": int(len(X_train)),
        "num_val": int(len(X_val)),
    }
    save_json(meta, out_dir / "meta.json")
    print(f"Saved: {out_dir/'train.npz'}, {out_dir/'val.npz'}")
    print("Metadata:", meta)

if __name__ == "__main__":
    main()