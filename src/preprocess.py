# src/preprocess.py
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle

def load_data(data_dir="data"):
    X = []
    y = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))])
    for idx, label in enumerate(classes):
        label_dir = os.path.join(data_dir, label)
        for fn in os.listdir(label_dir):
            if fn.endswith(".npy"):
                arr = np.load(os.path.join(label_dir, fn))
                X.append(arr)
                y.append(idx)
    X = np.vstack(X)
    y = np.array(y)
    return X, y, classes

def split_save(data_dir="data", out_path="processed_data.pkl", test_size=0.2, random_state=42):
    X, y, classes = load_data(data_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    with open(out_path, "wb") as f:
        pickle.dump({"X_train":X_train,"y_train":y_train,"X_val":X_val,"y_val":y_val,"classes":classes}, f)
    print("Saved processed data to", out_path)

if __name__ == "__main__":
    split_save()
