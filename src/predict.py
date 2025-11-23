# src/predict.py
import numpy as np
import tensorflow as tf
import os
import pickle

def load_model_and_classes(model_path="models/model.h5", processed_data="processed_data.pkl"):
    model = tf.keras.models.load_model(model_path)
    with open(processed_data, "rb") as f:
        data = pickle.load(f)
    classes = data["classes"]
    return model, classes

def predict_landmark(model, classes, landmark_vector):
    """
    landmark_vector: 1D numpy array (shape matching model input)
    returns: (pred_label, confidence)
    """
    x = np.expand_dims(landmark_vector, axis=0).astype(np.float32)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), probs
