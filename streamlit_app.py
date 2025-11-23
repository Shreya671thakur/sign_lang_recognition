# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from src.utils import mediapipe_hands, extract_hand_landmarks
from src.predict import load_model_and_classes, predict_landmark
import time
import os

st.set_page_config(page_title="Sign Language Recognition", layout="wide")
st.title("Sign Language Recognition — Live (MediaPipe + Keras)")

MODEL_PATH = "models/model.h5"
PROCESSED_DATA = "processed_data.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning("No model found. Train a model first (see README).")
    st.stop()

model, classes = load_model_and_classes(MODEL_PATH, PROCESSED_DATA)

run = st.checkbox("Run webcam")
confidence_threshold = st.slider("Confidence threshold", 0.1, 0.99, 0.6)

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    hands = mediapipe_hands()
    last_prediction = ("", 0.0)
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture from webcam.")
            break
        res = extract_hand_landmarks(frame, hands)
        display_frame = frame.copy()
        if res is not None:
            landmarks, drawing, hand_lms = res
            # draw landmarks
            drawing.draw_landmarks(display_frame, hand_lms, None)
            label, conf, probs = predict_landmark(model, classes, landmarks)
            if conf >= confidence_threshold:
                last_prediction = (label, conf)
            # show on frame
            cv2.putText(display_frame, f"{label} {conf:.2f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        else:
            cv2.putText(display_frame, "No hand", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # convert BGR to RGB for display in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        # must break streamlit loop from browser (press stop)
        if not st.session_state.get("run", True):
            break
        # small sleep to reduce CPU usage
        time.sleep(0.01)
    cap.release()
else:
    st.info("Enable webcam to start inference.")
