"""
capture_dataset.py
Session 1 tool: capture and auto-crop labeled coin images for dataset building.

Place ONE coin at a time in the webcam view. When the green detection ring
appears, press the matching denomination key to save the crop:

    1 = 25-Centavo   2 = 1-Piso   3 = 5-Piso   4 = 10-Piso   5 = 20-Piso
    q = Quit

Each keypress saves:
  - Cropped coin image → ngc_coin_dataset/dataset/<split>/<class>/<name>.jpg
  - Full raw frame     → ngc_coin_dataset/raw_frames/raw_<class>_<n>.jpg

The train/val/test split is chosen randomly at 70 / 15 / 15.
Counters resume from the last saved index so re-runs never overwrite images.
"""

import cv2
import os
import random
import sys

from config import (
    DATA_DIR, RAW_FRAMES_DIR, SPLITS, SPLIT_WEIGHTS, COIN_SPECS,
    HOUGH_DP, HOUGH_MIN_DIST_PX, HOUGH_PARAM1, HOUGH_PARAM2,
    HOUGH_MIN_RADIUS_PX, HOUGH_MAX_RADIUS_PX,
    CAMERA_INDEX,
)

# ---------------------------------------------------------------------------
# KEY MAPPING  — ascending PHP value (1 = cheapest → 5 = most expensive)
# ---------------------------------------------------------------------------
# key → (folder_name, display_label)
KEYMAP = {
    ord('1'): ("25_centavo", "25-Centavo"),
    ord('2'): ("1_piso",     "1-Piso"),
    ord('3'): ("5_piso",     "5-Piso"),
    ord('4'): ("10_piso",    "10-Piso"),
    ord('5'): ("20_piso",    "20-Piso"),
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def ensure_folders():
    """Create every required dataset subdirectory and the raw_frames folder."""
    os.makedirs(RAW_FRAMES_DIR, exist_ok=True)
    for folder_name, *_ in COIN_SPECS:
        for split in SPLITS:
            os.makedirs(os.path.join(DATA_DIR, split, folder_name), exist_ok=True)


def init_counters():
    """
    Scan existing saved files to find the highest used index per class
    so that resumed sessions append instead of overwriting.

    Returns
    -------
    dict : folder_name → next integer index (starts at 1 when folder is empty)
    """
    counters = {}
    for folder_name, *_ in COIN_SPECS:
        max_idx = 0
        for split in SPLITS:
            split_dir = os.path.join(DATA_DIR, split, folder_name)
            if not os.path.isdir(split_dir):
                continue
            for fname in os.listdir(split_dir):
                if not fname.lower().endswith(".jpg"):
                    continue
                # Expected filename format: <folder_name>_<NNNN>.jpg
                stem = fname[len(folder_name) + 1:].replace(".jpg", "").replace(".JPG", "")
                try:
                    max_idx = max(max_idx, int(stem))
                except ValueError:
                    pass
        counters[folder_name] = max_idx + 1
    return counters


def pick_split():
    """Randomly choose train / val / test according to the configured weights."""
    return random.choices(SPLITS, weights=SPLIT_WEIGHTS, k=1)[0]


def detect_largest_circle(gray_blurred):
    """
    Run HoughCircles and return the circle with the largest radius,
    or None if nothing is detected.
    """
    circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST_PX,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS_PX, maxRadius=HOUGH_MAX_RADIUS_PX,
    )
    if circles is None:
        return None
    return max(circles[0], key=lambda c: c[2])   # largest radius = main coin


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ensure_folders()
    counters = init_counters()
    session_counts = {folder_name: 0 for folder_name, *_ in COIN_SPECS}

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open camera index {CAMERA_INDEX}. "
                 "Run list_cameras.py to find the correct index, "
                 "then update CAMERA_INDEX in config.py.")

    flash_msg    = ""
    flash_frames = 0

    print("\nNGC Coin Dataset Capture")
    print("  Keys: 1=25-Centavo  2=1-Piso  3=5-Piso  4=10-Piso  5=20-Piso  q=Quit")
    print("  Next indices per class:", {k: v for k, v in counters.items()})
    print()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: Failed to read frame from webcam — retrying…")
            continue

        # --- Detection ---
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circle  = detect_largest_circle(blurred)

        display = frame.copy()
        crop    = None

        if circle is not None:
            x, y, r = circle.astype(int)
            pad = int(r * 0.15)
            x1 = max(0, x - r - pad)
            y1 = max(0, y - r - pad)
            x2 = min(frame.shape[1], x + r + pad)
            y2 = min(frame.shape[0], y + r + pad)
            crop = frame[y1:y2, x1:x2]
            cv2.circle(display, (x, y), r, (0, 255, 0), 2)
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)

        # --- Header bar ---
        cv2.rectangle(display, (0, 0), (display.shape[1], 38), (25, 25, 25), -1)
        cv2.putText(display, "1:25c  2:P1  3:P5  4:P10  5:P20  |  q: quit",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)

        # --- Session counts (top-right) ---
        y_off = 55
        for folder_name, label, *_ in COIN_SPECS:
            text = f"{label}: {session_counts[folder_name]}"
            cv2.putText(display, text,
                        (display.shape[1] - 165, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 120), 1)
            y_off += 22

        # --- Flash message on save ---
        if flash_frames > 0:
            cv2.putText(display, flash_msg,
                        (8, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
            flash_frames -= 1

        cv2.imshow("NGC Dataset Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key in KEYMAP and crop is not None and crop.size > 0:
            folder_name, label = KEYMAP[key]
            split = pick_split()
            idx   = counters[folder_name]
            counters[folder_name]       += 1
            session_counts[folder_name] += 1

            # Save cropped coin image
            fname     = f"{folder_name}_{idx:04d}.jpg"
            crop_path = os.path.join(DATA_DIR, split, folder_name, fname)
            cv2.imwrite(crop_path, crop)

            # Save full raw frame (for reference / re-cropping)
            raw_fname = f"raw_{folder_name}_{idx:04d}.jpg"
            raw_path  = os.path.join(RAW_FRAMES_DIR, raw_fname)
            cv2.imwrite(raw_path, frame)

            flash_msg    = f"Saved {label} → {split}/  [{idx:04d}]"
            flash_frames = 45
            print(f"  + {crop_path}")

    cap.release()
    cv2.destroyAllWindows()

    # --- Session summary ---
    print("\n--- Capture Session Summary ---")
    grand_total = 0
    for folder_name, label, *_ in COIN_SPECS:
        total_on_disk = sum(
            len([f for f in os.listdir(os.path.join(DATA_DIR, split, folder_name))
                 if f.lower().endswith(".jpg")])
            for split in SPLITS
            if os.path.isdir(os.path.join(DATA_DIR, split, folder_name))
        )
        print(f"  {label:12s}: {session_counts[folder_name]:3d} new  "
              f"({total_on_disk} total on disk)")
        grand_total += total_on_disk
    print(f"  {'TOTAL':12s}: {grand_total} images across all classes")


if __name__ == "__main__":
    main()
