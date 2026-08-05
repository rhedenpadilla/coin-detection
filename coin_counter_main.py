"""
coin_counter_main.py
Real-time Philippine NGC coin recognition and persistent value accumulator.

Workflow
--------
1. Place ONE coin at a time in the webcam view.
2. A green ring appears when a coin is detected; its predicted label is shown.
3. Press SPACE to confirm and add that coin's value to the running total.
4. Remove the coin and place the next one.
5. Repeat until done.

Controls
--------
  SPACE  — add currently detected coin to the total
  u      — undo the last added coin
  r      — reset the total and history to zero
  q      — quit

Classification
--------------
  Default  → CNN (MobileNetV2) loaded from ngc_coin_dataset/models/coin_classifier.h5
  Fallback → heuristic size + HSV-color classification (Option A) when the
             model file is not found or CNN confidence is below threshold.
"""

import cv2
import numpy as np
import os

from config import (
    CALIBRATION_SCALE_MM_PER_PX, COIN_SPECS,
    DIAMETER_TOLERANCE_MM,
    GOLD_HSV_LOWER, GOLD_HSV_UPPER, GOLD_RING_MIN_FRACTION,
    HOUGH_DP, HOUGH_MIN_DIST_PX, HOUGH_MIN_RADIUS_PX, HOUGH_MAX_RADIUS_PX,
    HOUGH_PARAM1, HOUGH_PARAM2,
    MODEL_PATH, CAMERA_INDEX,
)
from train_classifier import classify_with_cnn, load_cnn_model


# ---------------------------------------------------------------------------
# OPTION A: HEURISTIC CLASSIFICATION (calibrated diameter + HSV gold ring)
# ---------------------------------------------------------------------------

def _has_gold_ring(frame_bgr, x, y, r):
    """Return True when the outer annular region of the coin ROI is gold-toned."""
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(w, x + r), min(h, y + r)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    # Build an annular mask: full circle minus the inner 55 % radius
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cx, cy = x - x1, y - y1
    cv2.circle(mask, (cx, cy), r,            255, -1)
    cv2.circle(mask, (cx, cy), int(r * 0.55), 0,  -1)

    hsv        = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gold_mask  = cv2.inRange(hsv, GOLD_HSV_LOWER, GOLD_HSV_UPPER)
    ring_px    = cv2.countNonZero(mask)
    gold_px    = cv2.countNonZero(cv2.bitwise_and(gold_mask, mask))

    if ring_px == 0:
        return False
    return (gold_px / ring_px) >= GOLD_RING_MIN_FRACTION


def classify_by_size_and_color(frame_bgr, x, y, r):
    """
    Classify a coin using its measured diameter and (for 20-Piso) gold ring.

    Returns
    -------
    (label : str, value_php : float) or (None, 0.0) if no confident match.
    """
    diameter_mm = 2.0 * r * CALIBRATION_SCALE_MM_PER_PX
    candidates  = [
        spec for spec in COIN_SPECS
        if abs(diameter_mm - spec[3]) <= DIAMETER_TOLERANCE_MM
    ]
    if not candidates:
        return None, 0.0

    largest = max(candidates, key=lambda s: s[3])
    if largest[4]:   # is_bimetallic → must confirm with gold-ring test
        if _has_gold_ring(frame_bgr, x, y, r):
            return largest[1], largest[2]
        # Gold ring not found → exclude bimetallic and keep remaining candidates
        candidates = [c for c in candidates if not c[4]]
        if not candidates:
            return None, 0.0

    best = min(candidates, key=lambda s: abs(diameter_mm - s[3]))
    return best[1], best[2]


# ---------------------------------------------------------------------------
# CIRCLE DETECTION
# ---------------------------------------------------------------------------

def detect_coins(frame_bgr):
    """
    Detect circular objects in the frame using HoughCircles.

    Returns
    -------
    list of (x, y, r) integer tuples, one per detected circle.
    """
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST_PX,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS_PX, maxRadius=HOUGH_MAX_RADIUS_PX,
    )
    if circles is None:
        return []
    return [(int(x), int(y), int(r)) for x, y, r in np.round(circles[0])]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # ---- Determine classification mode ----
    cnn_ready = os.path.isfile(MODEL_PATH)
    if cnn_ready:
        load_cnn_model()   # pre-load into the module-level cache
        mode_label = "CNN (MobileNetV2)"
    else:
        mode_label = "Heuristic (size + color)"
        print(f"[INFO] Model not found at:\n  {MODEL_PATH}")
        print("[INFO] Using heuristic classification (Option A).")
        print("[INFO] Run train_classifier.py after building a dataset to enable CNN mode.\n")

    if CALIBRATION_SCALE_MM_PER_PX == 0.20:
        print("[WARNING] CALIBRATION_SCALE_MM_PER_PX is still the placeholder value (0.20 mm/px).")
        print("[WARNING] Heuristic diameter matching may be inaccurate. Edit config.py after calibration.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Run list_cameras.py to find the correct index, "
            "then update CAMERA_INDEX in config.py."
        )

    # ---- Persistent accumulator state ----
    total_value  = 0.0
    coin_history = []   # list of (label, value_php); used for undo + on-screen log

    flash_msg    = ""
    flash_frames = 0

    print(f"Mode: {mode_label}")
    print("Controls: SPACE = add  |  u = undo  |  r = reset  |  q = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: Failed to read webcam frame — retrying…")
            continue

        circles = detect_coins(frame)

        # Take the largest detected circle as the coin-in-view
        current_label, current_value = None, 0.0
        if circles:
            x, y, r = max(circles, key=lambda c: c[2])

            # Try CNN first (if model is loaded)
            if cnn_ready:
                current_label, current_value = classify_with_cnn(frame, x, y, r)

            # Fallback to heuristics when CNN is absent or returns low-confidence
            if current_label is None:
                current_label, current_value = classify_by_size_and_color(frame, x, y, r)

            # Draw detection ring
            ring_color = (0, 255, 0) if current_label else (0, 0, 255)
            cv2.circle(frame, (x, y), r, ring_color, 2)
            cv2.circle(frame, (x, y), 5, ring_color, -1)
            lbl_text = current_label if current_label else "?"
            cv2.putText(frame, lbl_text,
                        (x - r, y - r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, ring_color, 2)

        # ---- Header: running total ----
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(frame, f"TOTAL:  PHP {total_value:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Mode label (top-right)
        mode_short = "CNN" if cnn_ready else "Heuristic"
        cv2.putText(frame, mode_short,
                    (frame.shape[1] - 115, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1)

        # ---- Last-added log (up to 6 entries) ----
        y_off = 58
        cv2.putText(frame, "Added:", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1)
        y_off += 18
        for lbl, val in reversed(coin_history[-6:]):
            cv2.putText(frame, f"  {lbl}   +PHP {val:.2f}",
                        (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 255, 160), 1)
            y_off += 17

        # ---- Bottom prompt ----
        detected_str = current_label if current_label else "no coin"
        cv2.putText(frame,
                    f"SPACE: add [{detected_str}]   u: undo   r: reset   q: quit",
                    (8, frame.shape[0] - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)

        # ---- Flash message ----
        if flash_frames > 0:
            cv2.putText(frame, flash_msg,
                        (8, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 130), 2)
            flash_frames -= 1

        cv2.imshow("NGC Coin Counter", frame)
        key = cv2.waitKey(1) & 0xFF

        # ---- Controls ----
        if key == ord('q'):
            break

        elif key == ord(' '):   # SPACE — add current detection
            if current_label is not None:
                total_value += current_value
                coin_history.append((current_label, current_value))
                flash_msg    = f"+ {current_label}   PHP {current_value:.2f}"
                flash_frames = 45
                print(f"  + {current_label}  PHP {current_value:.2f}  |  "
                      f"Total: PHP {total_value:.2f}")
            else:
                flash_msg    = "No coin detected — move coin into view first"
                flash_frames = 40

        elif key == ord('u'):   # undo last
            if coin_history:
                lbl, val    = coin_history.pop()
                total_value = max(0.0, total_value - val)
                flash_msg   = f"Undo: -{lbl}   -PHP {val:.2f}"
                flash_frames = 45
                print(f"  Undo {lbl}  |  Total: PHP {total_value:.2f}")
            else:
                flash_msg    = "Nothing to undo"
                flash_frames = 30

        elif key == ord('r'):   # reset everything
            total_value  = 0.0
            coin_history.clear()
            flash_msg    = "Reset — Total: PHP 0.00"
            flash_frames = 45
            print("  Reset total.")

    cap.release()
    cv2.destroyAllWindows()

    # ---- Final summary ----
    print(f"\n--- Final Total: PHP {total_value:.2f} ---")
    if coin_history:
        print("Coins added this session:")
        for lbl, val in coin_history:
            print(f"  {lbl:15s}  PHP {val:.2f}")
    else:
        print("No coins were added.")


if __name__ == "__main__":
    main()
