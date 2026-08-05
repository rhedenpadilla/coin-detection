"""
config.py
Shared constants and paths for the NGC coin detection system.

All path constants are derived from this file's own location (PROJECT_ROOT)
so the project works correctly regardless of the shell's current working
directory.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# ROOT PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT   = os.path.join(PROJECT_ROOT, "ngc_coin_dataset")

DATA_DIR       = os.path.join(DATASET_ROOT, "dataset")
RAW_FRAMES_DIR = os.path.join(DATASET_ROOT, "raw_frames")
MODELS_DIR     = os.path.join(DATASET_ROOT, "models")
MODEL_PATH     = os.path.join(MODELS_DIR, "coin_classifier.h5")

# ---------------------------------------------------------------------------
# COIN SPECIFICATIONS
# (folder_name, display_label, value_php, diameter_mm, is_bimetallic)
# ---------------------------------------------------------------------------
COIN_SPECS = [
    ("25_centavo", "25-Centavo",  0.25, 20.0, False),
    ("1_piso",     "1-Piso",      1.00, 24.0, False),
    ("5_piso",     "5-Piso",      5.00, 25.0, False),
    ("10_piso",    "10-Piso",    10.00, 27.0, False),
    ("20_piso",    "20-Piso",    20.00, 30.0, True),
]

SPLITS        = ["train", "val", "test"]
SPLIT_WEIGHTS = [0.70, 0.15, 0.15]

# ---------------------------------------------------------------------------
# CNN INFERENCE
# ---------------------------------------------------------------------------
IMG_SIZE   = (96, 96)    # (height, width) — must match training
BATCH_SIZE = 16
EPOCHS     = 25
NUM_CLASSES = 5

# Keras flow_from_directory sorts class folders alphabetically.
# Folder names sorted: 1_piso, 10_piso, 20_piso, 25_centavo, 5_piso
# Each entry: index → (display_label, value_php)
CNN_CLASS_MAP = [
    ("1-Piso",      1.00),   # index 0  ← folder "1_piso"
    ("10-Piso",    10.00),   # index 1  ← folder "10_piso"
    ("20-Piso",    20.00),   # index 2  ← folder "20_piso"
    ("25-Centavo",  0.25),   # index 3  ← folder "25_centavo"
    ("5-Piso",      5.00),   # index 4  ← folder "5_piso"
]

CONFIDENCE_THRESHOLD = 0.60   # minimum softmax probability to accept a prediction

# ---------------------------------------------------------------------------
# CAMERA
# ---------------------------------------------------------------------------
# 0 = built-in / USB webcam (default)
# 1, 2, … = other cameras (e.g. Phone Link virtual camera)
# Run list_cameras.py to discover which index your Phone Link camera is.
CAMERA_INDEX = 1

# ---------------------------------------------------------------------------
# HOUGH CIRCLE DETECTION (shared by capture and main scripts)
# ---------------------------------------------------------------------------
HOUGH_DP           = 1.2
HOUGH_MIN_DIST_PX  = 60
HOUGH_PARAM1       = 100
HOUGH_PARAM2       = 40
HOUGH_MIN_RADIUS_PX = 25
HOUGH_MAX_RADIUS_PX = 160

# ---------------------------------------------------------------------------
# OPTION A: HEURISTIC CLASSIFICATION (size + HSV color)
# ---------------------------------------------------------------------------
# mm per pixel from a one-time calibration. Update after measuring!
CALIBRATION_SCALE_MM_PER_PX = 0.20   # placeholder

DIAMETER_TOLERANCE_MM = 1.3

# HSV range for the gold outer ring of the 20-Piso bimetallic coin
GOLD_HSV_LOWER       = np.array([15,  60,  90])
GOLD_HSV_UPPER       = np.array([35, 255, 255])
GOLD_RING_MIN_FRACTION = 0.10   # min gold-pixel fraction in the outer ring
