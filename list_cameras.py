"""
list_cameras.py
Utility to discover all camera devices available to OpenCV on this system.
Run this to find which index corresponds to your Phone Link virtual camera.

Usage:
    python list_cameras.py

Output example:
    [0] Built-in / USB webcam     ← index 0
    [1] SAMSUNG Galaxy S24 Ultra  ← Phone Link camera  (use CAMERA_INDEX = 1)
    [2] (nothing — search stopped)

After finding the correct index, edit config.py:
    CAMERA_INDEX = 1   ← replace with your Phone Link camera index
"""

import cv2

MAX_INDEX_TO_TEST = 6   # test indices 0–5; increase if needed

def list_cameras():
    print("\nScanning for available cameras…\n")
    found = []

    for idx in range(MAX_INDEX_TO_TEST):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # CAP_DSHOW = DirectShow (Windows)
        if not cap.isOpened():
            print(f"  [{idx}]  — not available")
            cap.release()
            continue

        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  [{idx}]  ✓ Camera found  ({w}x{h})")
            found.append(idx)

            # Show a quick preview window for 2 seconds so you can identify it visually
            cv2.imshow(f"Camera [{idx}] — press any key", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        else:
            print(f"  [{idx}]  — opened but could not read frame")

        cap.release()

    print()
    if not found:
        print("No cameras found. Check that Phone Link is connected and camera sharing is enabled.")
    else:
        print(f"Found {len(found)} camera(s) at index(es): {found}")
        print()
        print("To use your Phone Link camera, edit config.py:")
        print(f"  CAMERA_INDEX = {found[-1] if len(found) > 1 else found[0]}")
        print("  (use whichever index showed your phone's view)")


if __name__ == "__main__":
    list_cameras()
