
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from utils import set_seed, ensure_dir, preprocess_resnet50, load_json, compute_class_weights

def build_model(input_shape=(224, 224, 3), num_classes=2, train_base=False):
    base = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = train_base  # freeze for quick finetuning; set True to unfreeze
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    else:
        out = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss, metrics=metrics)
    return model

def npz_to_dataset(images_uint8, labels_int, batch_size, shuffle=True, augment=True):
    X = preprocess_resnet50(images_uint8)
    y = labels_int
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42, reshuffle_each_iteration=True)
    # Light augmentations (optional)
    if augment:
        aug = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
            ]
        )
        ds = ds.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_base", action="store_true", help="Unfreeze backbone")
    args = parser.parse_args()

    set_seed(42)
    ensure_dir(args.out_dir)

    train_npz = np.load(Path(args.data_npz_dir) / "train.npz")
    val_npz = np.load(Path(args.data_npz_dir) / "val.npz")
    meta = load_json(Path(args.data_npz_dir) / "meta.json")

    X_tr, y_tr = train_npz["images"], train_npz["labels"]
    X_va, y_va = val_npz["images"], val_npz["labels"]
    num_classes = len(meta["class_to_idx"])

    train_ds = npz_to_dataset(X_tr, y_tr, args.batch_size, shuffle=True, augment=True)
    val_ds = npz_to_dataset(X_va, y_va, args.batch_size, shuffle=False, augment=False)

    model = build_model(input_shape=(meta["img_size"], meta["img_size"], 3),
                        num_classes=num_classes, train_base=args.train_base)

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(str(Path(args.out_dir) / "resnet50_best.keras"),
                                           monitor="val_auc" if num_classes == 2 else "val_accuracy",
                                           mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.CSVLogger(str(Path(args.out_dir) / "training_log.csv")),
    ]

    class_weights = compute_class_weights(y_tr)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cbs,
        class_weight=class_weights if num_classes == 2 else None,
        verbose=1,
    )

    model.save(Path(args.out_dir) / "resnet50_final.keras")
    print("Training complete. Saved best & final models in:", args.out_dir)

if __name__ == "__main__":
    main()