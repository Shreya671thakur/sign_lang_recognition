# src/utils.py
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_hands():
    # static_image_mode=False (video), max_num_hands=1 for single-hand system
    return mp_hands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.6,
                          min_tracking_confidence=0.6)

def extract_hand_landmarks(frame, hands):
    """
    Input: BGR frame (OpenCV)
    Output: flattened landmarks vector (x,y) normalized relative to bounding box; or None
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = hands.process(image)
    image.flags.writeable = True

    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]  # single hand
    # gather raw landmarks
    h, w, _ = frame.shape
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append([lm.x, lm.y, lm.z])
    pts = np.array(pts)  # (21,3)

    # Normalize: translate center to origin and scale by hand bbox (robust)
    xs = pts[:,0]
    ys = pts[:,1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    box_w = max_x - min_x
    box_h = max_y - min_y
    box_size = max(box_w, box_h)
    if box_size < 1e-6:
        return None
    # shift to center
    cx = (max_x + min_x) / 2.0
    cy = (max_y + min_y) / 2.0
    normalized = []
    for (x,y,z) in pts:
        nx = (x - cx) / box_size
        ny = (y - cy) / box_size
        nz = z  # optional
        normalized.append([nx, ny, nz])
    normalized = np.array(normalized).flatten()  # shape (63,)
    return normalized, mp_drawing, hand_landmarks
