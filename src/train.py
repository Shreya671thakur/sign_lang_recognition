# src/train.py
import argparse
import os
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train(data_path="processed_data.pkl", epochs=50, batch_size=32, model_out="models/model.h5"):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    classes = data["classes"]
    print("Classes:", classes)
    input_shape = X_train.shape[1]
    num_classes = len(classes)
    model = build_model(input_shape, num_classes)
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_out, save_best_only=True, monitor="val_accuracy", mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    # evaluate
    preds = model.predict(X_val).argmax(axis=1)
    print("Val accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds, target_names=classes, zero_division=0))
    print("Saved model to", model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="processed_data.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_out", default="models/model.h5")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    train(args.data_path, args.epochs, args.batch_size, args.model_out)
