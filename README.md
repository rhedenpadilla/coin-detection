# 🪙 NGC Philippine Coin Detection & Counter

A real-time computer-vision system that detects, classifies, and accumulates the total value of **Philippine NGC coins** (New Generation Currency) using a webcam feed. The system can run in two classification modes: a fast **heuristic** mode (no training needed) or a more accurate **CNN/deep-learning** mode powered by a fine-tuned MobileNetV2 model.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Supported Coins](#supported-coins)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage — Step-by-Step Workflow](#usage--step-by-step-workflow)
  - [Step 0 — Find Your Camera Index](#step-0--find-your-camera-index)
  - [Step 1 — Build the Dataset (Optional, for CNN mode)](#step-1--build-the-dataset-optional-for-cnn-mode)
  - [Step 2 — Train the CNN Classifier (Optional)](#step-2--train-the-cnn-classifier-optional)
  - [Step 3 — Run the Coin Counter](#step-3--run-the-coin-counter)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Classification Modes](#classification-modes)
- [Calibration Guide](#calibration-guide)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)

---

## Overview

Place coins one at a time in front of your webcam. The system:

1. **Detects** circular shapes in the live video using the Hough Circle Transform.
2. **Classifies** the coin using either a CNN model (MobileNetV2) or a heuristic based on measured diameter and gold-ring color detection.
3. **Accumulates** the total PHP value as you confirm each coin with the spacebar.

The system was designed to work with a **Phone Link virtual camera** (your phone's camera streamed to Windows) for a better image, but a standard USB or built-in webcam also works.

---

## Supported Coins

| Key | Denomination | Value (PHP) | Diameter | Notes |
|-----|-------------|-------------|----------|-------|
| `1` | 25-Centavo  | ₱0.25       | 20.0 mm  | Silver-tone |
| `2` | 1-Piso      | ₱1.00       | 24.0 mm  | Silver-tone |
| `3` | 5-Piso      | ₱5.00       | 25.0 mm  | Silver-tone |
| `4` | 10-Piso     | ₱10.00      | 27.0 mm  | Silver-tone |
| `5` | 20-Piso     | ₱20.00      | 30.0 mm  | **Bimetallic** (gold outer ring) |

> The key column applies only during **dataset capture** (`capture_dataset.py`). During the coin counter, classification is automatic.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Webcam / Phone Camera                    │
└────────────────────────────┬────────────────────────────────────┘
                             │  Live BGR frames
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Hough Circle Transform (coin_counter_main.py)      │
│        Gaussian blur → HoughCircles → largest circle = coin     │
└────────────────────────────┬────────────────────────────────────┘
                             │  (x, y, radius)
                 ┌───────────┴────────────┐
                 ▼                        ▼
   ┌─────────────────────┐    ┌─────────────────────────────┐
   │  CNN Classifier     │    │  Heuristic Classifier       │
   │  (MobileNetV2)      │    │  diameter + HSV gold ring   │
   │  (primary, if       │    │  (fallback / no model file) │
   │   model exists)     │    │                             │
   └─────────┬───────────┘    └──────────────┬──────────────┘
             └──────────────┬────────────────┘
                            ▼
            ┌────────────────────────────────┐
            │  Running Total Accumulator     │
            │  SPACE: add | u: undo | r: reset│
            └────────────────────────────────┘
```

### Three-Session Pipeline (for CNN mode)

```
Session 1                Session 2               Session 3
capture_dataset.py  →  train_classifier.py  →  coin_counter_main.py
(take photos)           (train model)            (live counting)
```

---

## Prerequisites

### Hardware
- A **webcam** (built-in, USB, or Phone Link virtual camera on Windows)
- Recommended: **Windows 10/11** with [Phone Link](https://aka.ms/phone-link) for higher-quality camera input

### Software

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python**  | **3.11.x** | ⚠️ **Required.** TensorFlow does not support Python 3.12+. |
| pip         | Latest  | Bundled with Python |
| Git         | Any     | For cloning the repo |

> **Why Python 3.11 specifically?** TensorFlow 2.21 only supports Python up to 3.11. Using 3.12 or later will cause a hard install failure.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/coin-detection.git
cd coin-detection
```

### 2. Create a Python 3.11 virtual environment

```powershell
# Windows (PowerShell) — make sure py launcher resolves to 3.11
py -3.11 -m venv venv
```

### 3. Activate the virtual environment

```powershell
# Windows (PowerShell)
.\venv\Scripts\activate
```

You should see `(venv)` prepended to your shell prompt.

### 4. Install dependencies

```powershell
pip install -r requirements.txt --timeout 600
```

> ⚠️ **TensorFlow is ~350 MB.** A stable internet connection is required. If the download times out, retry with:
> ```powershell
> pip install tensorflow --timeout 600 --retries 10
> ```

### Installed packages

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | 2.21.0 | CNN training & inference (MobileNetV2) |
| `opencv-python` | 5.0.0.93 | Camera capture, Hough circles, image processing |
| `numpy` | 2.4.6 | Numerical operations |

---

## Usage — Step-by-Step Workflow

### Step 0 — Find Your Camera Index

If you have multiple cameras (e.g., a Phone Link virtual camera alongside a built-in webcam), run:

```powershell
python list_cameras.py
```

This scans indices 0–5, shows which cameras OpenCV can open, and briefly displays a preview so you can visually identify each one.

**Example output:**
```
[0]  ✓ Camera found  (1280x720)   ← built-in webcam
[1]  ✓ Camera found  (1920x1080)  ← Phone Link camera  (use this!)
[2]  — not available
```

Then open `config.py` and set:
```python
CAMERA_INDEX = 1   # ← your phone's index
```

---

### Step 1 — Build the Dataset (Optional, for CNN mode)

> **Skip this step** if you only want to use the fast heuristic mode (no model needed).

```powershell
python capture_dataset.py
```

**Controls during capture:**

| Key | Action |
|-----|--------|
| `1` | Save current view as **25-Centavo** |
| `2` | Save current view as **1-Piso** |
| `3` | Save current view as **5-Piso** |
| `4` | Save current view as **10-Piso** |
| `5` | Save current view as **20-Piso** |
| `q` | Quit |

**Tips:**
- Place **one coin at a time**; wait for the **green detection ring** to appear before pressing a key.
- Aim for **at least 50–100 images per class** for good accuracy.
- Vary the angle, lighting, and background to improve model robustness.
- Images are automatically split into **train / val / test** folders (70 / 15 / 15%).
- Re-running the script **resumes** from the last saved index — it will never overwrite existing images.

**Saved files per keypress:**
```
ngc_coin_dataset/
  dataset/
    train/  <class>/  <class>_NNNN.jpg    ← cropped coin image
    val/    <class>/  <class>_NNNN.jpg
    test/   <class>/  <class>_NNNN.jpg
  raw_frames/
    raw_<class>_NNNN.jpg                  ← full frame (for debugging)
```

---

### Step 2 — Train the CNN Classifier (Optional)

> **Skip this step** if you only want to use the fast heuristic mode.
> Requires the dataset from Step 1.

```powershell
python train_classifier.py
```

**What it does:**
1. Loads images from `ngc_coin_dataset/dataset/train/` and `val/`.
2. Applies data augmentation (rotations, flips, brightness shifts, zooms).
3. Fine-tunes a **MobileNetV2** backbone pretrained on ImageNet.
4. Trains for up to **25 epochs** with early stopping (patience = 6).
5. Saves:
   - `ngc_coin_dataset/models/coin_classifier.h5` — final model
   - `ngc_coin_dataset/models/checkpoint_best.h5` — best validation-accuracy checkpoint

**Training hyperparameters** (adjustable in `config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | `(96, 96)` | Input image size (height × width) |
| `BATCH_SIZE` | `16` | Samples per gradient step |
| `EPOCHS` | `25` | Maximum training epochs |
| `CONFIDENCE_THRESHOLD` | `0.60` | Minimum softmax confidence to accept a CNN prediction |

---

### Step 3 — Run the Coin Counter

```powershell
python coin_counter_main.py
```

**On-screen controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Add the currently detected coin to the running total |
| `u` | Undo the last added coin |
| `r` | Reset total and history to ₱0.00 |
| `q` | Quit and display final session summary |

**Workflow:**
1. Place **one coin** in the webcam view.
2. A **green ring** appears when a coin is detected; its predicted label is shown above it.
3. Press `SPACE` to confirm and accumulate its value.
4. Remove the coin, place the next one, and repeat.

**HUD elements:**
- **Top bar:** running PHP total and active classification mode (CNN or Heuristic).
- **Left column:** last 6 coins added (label + value).
- **Bottom bar:** controls reminder and currently detected denomination.

---

## Configuration

All tunable parameters live in [`config.py`](config.py). Edit this file before running any script.

### Key settings

```python
# ── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX = 1          # Run list_cameras.py to find the correct index

# ── Calibration (for heuristic mode) ────────────────────────────────────────
CALIBRATION_SCALE_MM_PER_PX = 0.20   # ⚠️ PLACEHOLDER — measure and update!

# ── CNN inference ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # CNN predictions below this fall back to heuristic

# ── Hough Circle Detection ───────────────────────────────────────────────────
HOUGH_DP            = 1.2
HOUGH_MIN_DIST_PX   = 60
HOUGH_PARAM1        = 100   # Canny edge upper threshold
HOUGH_PARAM2        = 40    # Accumulator threshold (lower = more detections)
HOUGH_MIN_RADIUS_PX = 25
HOUGH_MAX_RADIUS_PX = 160

# ── HSV gold-ring detection (20-Piso bimetallic) ─────────────────────────────
GOLD_HSV_LOWER         = [15,  60,  90]
GOLD_HSV_UPPER         = [35, 255, 255]
GOLD_RING_MIN_FRACTION = 0.10
```

---

## Project Structure

```
coin-detection/
│
├── config.py               # Shared constants, paths, and tunable parameters
├── list_cameras.py          # Utility: discover available camera indices
├── capture_dataset.py       # Session 1: build labeled image dataset
├── train_classifier.py      # Session 2: train MobileNetV2 CNN
├── coin_counter_main.py     # Session 3: live coin detection & value counter
│
├── requirements.txt         # Python dependencies
├── .gitignore
│
└── ngc_coin_dataset/        # Created at runtime (not committed to git)
    ├── dataset/
    │   ├── train/
    │   │   ├── 1_piso/
    │   │   ├── 5_piso/
    │   │   ├── 10_piso/
    │   │   ├── 20_piso/
    │   │   └── 25_centavo/
    │   ├── val/    (same structure)
    │   └── test/   (same structure)
    ├── raw_frames/          # Full webcam frames saved during capture
    └── models/
        ├── coin_classifier.h5      # Final trained model
        └── checkpoint_best.h5      # Best validation checkpoint
```

---

## Classification Modes

The counter automatically selects the best available mode at startup.

### Mode A — CNN (MobileNetV2) — *Primary*

- **Requires:** `ngc_coin_dataset/models/coin_classifier.h5` (trained in Session 2)
- **How it works:** Crops the detected coin ROI, resizes it to 96×96, and runs it through the trained neural network. The predicted class is accepted only if softmax confidence ≥ `CONFIDENCE_THRESHOLD` (0.60 by default).
- **Falls back to heuristic** when confidence is too low.

### Mode B — Heuristic (Size + HSV Color) — *Fallback*

- **Requires:** Accurate `CALIBRATION_SCALE_MM_PER_PX` value in `config.py`
- **How it works:**
  1. Converts the circle radius (pixels) to real-world diameter (mm) using the calibration scale.
  2. Matches the diameter against known coin specs within a ±1.3 mm tolerance.
  3. For the **20-Piso** (bimetallic), additionally checks for a gold-toned outer ring using HSV color analysis in an annular mask (outer 45% of radius).
- **Active when:** The model file is not found, or CNN confidence is below threshold.

---

## Calibration Guide

The heuristic mode requires knowing how many millimetres correspond to one pixel at your camera's working distance. To calibrate:

1. Place a **coin of known diameter** (e.g., the 10-Piso coin = 27 mm) in the webcam view.
2. Note the detected radius in pixels (printed in the terminal or measured in the preview).
3. Calculate the scale:
   ```
   scale = known_diameter_mm / (2 × detected_radius_px)
   ```
4. Update `config.py`:
   ```python
   CALIBRATION_SCALE_MM_PER_PX = 0.XX   # replace with your measured value
   ```

> ⚠️ **Recalibrate whenever** you change the camera, adjust the zoom level, or move the camera closer/further from the coins.

---

## Troubleshooting

### ❌ `Could not open camera index N`
- Run `python list_cameras.py` and update `CAMERA_INDEX` in `config.py`.
- Ensure Phone Link is connected and camera sharing is enabled:
  - **Settings → Cross-device experience → Phone Link → Camera**

### ❌ TensorFlow install times out
```powershell
pip install tensorflow --timeout 600 --retries 10
```
Ensure you are on a stable internet connection; TensorFlow is ~350 MB.

### ❌ No coins detected (no green ring)
Adjust the Hough parameters in `config.py`:
- **Decrease `HOUGH_PARAM2`** (e.g., 40 → 30) to detect more circles.
- **Adjust `HOUGH_MIN_RADIUS_PX` / `HOUGH_MAX_RADIUS_PX`** to match the coin size in your view.
- Improve **lighting** — even, diffuse light reduces shadows that confuse edge detection.

### ❌ Wrong denomination detected (heuristic mode)
- Recalibrate `CALIBRATION_SCALE_MM_PER_PX` (see [Calibration Guide](#calibration-guide)).
- The 5-Piso (25 mm) and 1-Piso (24 mm) differ by only 1 mm; accurate calibration is essential.

### ❌ Low CNN accuracy
- Capture more images per class (aim for 100+).
- Improve lighting consistency between capture and inference sessions.
- Retrain with `python train_classifier.py` after adding more images.

### ❌ `ModuleNotFoundError: No module named 'tensorflow'`
Ensure the virtual environment is activated:
```powershell
.\venv\Scripts\activate
```

---

## Known Limitations

- **One coin at a time.** The counter is designed for sequential single-coin placement. Only the **largest** detected circle is used; multiple coins in view simultaneously may cause misclassification.
- **Lighting sensitivity.** Very bright backgrounds or strong shadows may confuse the Hough Circle Transform or the HSV gold-ring detector.
- **Fixed working distance.** The heuristic mode's accuracy is tied to the calibration performed at a specific camera-to-coin distance.
- **Windows-first.** `list_cameras.py` uses DirectShow (`cv2.CAP_DSHOW`), which is Windows-specific. On Linux/macOS, remove the `cv2.CAP_DSHOW` flag from `list_cameras.py`.
- **No fractional-centavo rounding.** The running total is a floating-point sum; in rare edge cases, IEEE-754 rounding may cause sub-centavo display artifacts (e.g., ₱1.4999…). This does not affect the final printed summary.

---

## License

This project is for educational and personal use. No license is currently specified — please contact the author before distributing or commercializing.
