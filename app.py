import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model("model/asl_model.h5")
labels = open("model/labels.txt").read().splitlines()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

st.title("ASL Sign Language Recognition")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img)
    prediction = ""

    if results.multi_hand_landmarks:
        lm = []
        for i in results.multi_hand_landmarks[0].landmark:
            lm.extend([i.x, i.y, i.z])

        lm = np.array(lm).reshape(1, -1)
        pred = model.predict(lm)
        prediction = labels[np.argmax(pred)]

    cv2.putText(frame, prediction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 2)

    FRAME_WINDOW.image(frame)

cap.release()