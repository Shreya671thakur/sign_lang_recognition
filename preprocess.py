import os
import cv2
import numpy as np
import mediapipe as mp

DATASET_DIR = "dataset"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks).flatten()

print("Processing dataset...")

for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    save_folder = os.path.join(OUTPUT_DIR, label)
    os.makedirs(save_folder, exist_ok=True)

    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        lm = extract_landmarks(img_path)

        if lm is not None:
            save_path = os.path.join(save_folder, img_file.replace(".jpg", ".npy"))
            np.save(save_path, lm)

print("DONE! All landmarks saved to /processed_data")