import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

DATA_DIR = "processed_data"

X, y = [], []
labels = sorted(os.listdir(DATA_DIR))
label_to_index = {label: i for i, label in enumerate(labels)}

for label in labels:
    path = os.path.join(DATA_DIR, label)
    for file in os.listdir(path):
        data = np.load(os.path.join(path, file))
        X.append(data)
        y.append(label_to_index[label])

X = np.array(X)
y = to_categorical(y, num_classes=len(labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, callbacks=[es])

os.makedirs("model", exist_ok=True)
model.save("model/asl_model.h5")

with open("model/labels.txt", "w") as f:
    for l in labels:
        f.write(l + "\n")

print("Training complete! Model saved to /model")