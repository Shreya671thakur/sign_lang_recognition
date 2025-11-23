# src/collect_data.py
import argparse
import os
import time
import numpy as np
import cv2
from src.utils import mediapipe_hands, extract_hand_landmarks

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collect(label: str, count: int, out_dir: str):
    label_dir = os.path.join(out_dir, label)
    ensure_dir(label_dir)
    cap = cv2.VideoCapture(0)
    hands = mediapipe_hands()
    captured = 0
    print(f"[INFO] Starting capture for label '{label}'. Press 'q' to quit early.")
    time.sleep(1.0)
    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break
        res = extract_hand_landmarks(frame, hands)
        if res is None:
            cv2.putText(frame, "No hand detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.imshow("Collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue
        landmarks, drawing, hand_landmarks = res
        # show drawing
        drawing.draw_landmarks(frame, hand_landmarks, None)
        cv2.putText(frame, f"Captured: {captured}/{count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        cv2.imshow("Collect", frame)
        # Save landmark
        fname = os.path.join(label_dir, f"{label}_{captured}.npy")
        np.save(fname, landmarks)
        captured += 1
        # short pause so user can change pose
        time.sleep(0.25)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Finished. Collected {captured} samples for label '{label}' into {label_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Label name (e.g., A, B, hello)")
    parser.add_argument("--count", type=int, default=200, help="Number of samples to collect")
    parser.add_argument("--out_dir", default="data", help="Output directory for data")
    args = parser.parse_args()
    collect(args.label, args.count, args.out_dir)
