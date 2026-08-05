"""
train_classifier.py
Session 2 tool: train a MobileNetV2-based CNN classifier on the coin
dataset built in Session 1.

Usage:
    python train_classifier.py

Output:
    ngc_coin_dataset/models/coin_classifier.h5      ← final model
    ngc_coin_dataset/models/checkpoint_best.h5      ← best checkpoint (by val_accuracy)

This module also exposes two importable helpers used by coin_counter_main.py:
    load_cnn_model()            → loads / returns the cached Keras model
    classify_with_cnn(frame, x, y, r) → (label, value_php) or (None, 0.0)
"""

import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    BATCH_SIZE, CNN_CLASS_MAP, CONFIDENCE_THRESHOLD,
    DATA_DIR, EPOCHS, IMG_SIZE, MODELS_DIR, MODEL_PATH, NUM_CLASSES,
)

# ---------------------------------------------------------------------------
# INFERENCE HELPERS  (importable by coin_counter_main.py)
# ---------------------------------------------------------------------------

_cnn_model = None   # lazy-loaded module-level singleton


def load_cnn_model():
    """
    Load the trained model from MODEL_PATH once and cache it for the
    process lifetime.  Returns None if the model file does not exist yet.
    """
    global _cnn_model
    if _cnn_model is None:
        if not os.path.isfile(MODEL_PATH):
            return None
        print(f"[train_classifier] Loading model from {MODEL_PATH} …")
        _cnn_model = tf.keras.models.load_model(MODEL_PATH)
    return _cnn_model


def classify_with_cnn(frame_bgr, x, y, r):
    """
    Classify a single coin ROI using the trained CNN.

    Parameters
    ----------
    frame_bgr : np.ndarray  Full BGR webcam frame.
    x, y, r   : int         Circle centre and radius in pixels.

    Returns
    -------
    (label : str, value_php : float)
        or (None, 0.0) when confidence < CONFIDENCE_THRESHOLD or model absent.
    """
    model = load_cnn_model()
    if model is None:
        return None, 0.0

    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(w, x + r), min(h, y + r)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0.0

    roi_resized = cv2.resize(roi, IMG_SIZE).astype("float32") / 255.0
    roi_batch   = np.expand_dims(roi_resized, axis=0)   # shape: (1, H, W, 3)

    preds      = model.predict(roi_batch, verbose=0)[0]
    idx        = int(np.argmax(preds))
    confidence = float(preds[idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return None, 0.0

    label, value = CNN_CLASS_MAP[idx]
    return label, value


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train():
    """
    Build, train, and save the MobileNetV2 coin classifier.

    Folder layout expected under DATA_DIR:
        dataset/
          train/  { 1_piso, 10_piso, 20_piso, 25_centavo, 5_piso }
          val/    { same }
    """
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    if not os.path.isdir(train_dir):
        sys.exit(
            f"ERROR: Training directory not found: {train_dir}\n"
            "Run capture_dataset.py first to build the dataset."
        )

    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Data generators ---
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=180,       # coins have no fixed orientation
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    # --- Verify class-index mapping against CNN_CLASS_MAP ---
    print("\nKeras class index mapping (must match CNN_CLASS_MAP in config.py):")
    all_match = True
    for folder_name, keras_idx in sorted(train_gen.class_indices.items(),
                                          key=lambda kv: kv[1]):
        expected_label, expected_value = CNN_CLASS_MAP[keras_idx]
        print(f"  [{keras_idx}] {folder_name:15s} → {expected_label}  PHP {expected_value:.2f}")
        # folder_name (e.g. "1_piso") should map to the display label ("1-Piso")
        if folder_name.replace("_", "-").lower() not in expected_label.lower():
            print(f"        ^^^ WARNING: folder/label mismatch — check CNN_CLASS_MAP in config.py")
            all_match = False
    if all_match:
        print("  ✓ All mappings look consistent.\n")

    # --- Build model (transfer learning from MobileNetV2) ---
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # freeze pretrained backbone

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # --- Callbacks ---
    checkpoint_path = os.path.join(MODELS_DIR, "checkpoint_best.h5")
    cb_list = [
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # --- Train ---
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=cb_list,
    )

    # --- Save final model ---
    model.save(MODEL_PATH)
    print(f"\nModel saved  →  {MODEL_PATH}")
    print(f"Best checkpoint  →  {checkpoint_path}")

    return history


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
