import os
import numpy as np
import tensorflow as tf
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Base directory = d:\Assesment  (two levels up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "Data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ==========================================
# LOAD PREPROCESSED DATA
# ==========================================

X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))

Y_train = np.load(os.path.join(PROCESSED_DIR, "Y_train.npy"))
Y_test  = np.load(os.path.join(PROCESSED_DIR, "Y_test.npy"))

print("\nDataset Loaded Successfully")

print("X_train Shape:", X_train.shape)
print("Y_train Shape:", Y_train.shape)

# ==========================================
# BUILD LSTM MODEL
# ==========================================

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dense(32, activation='relu'))

model.add(Dense(2))

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer=Adam(0.001),
    loss='mse',
    metrics=['mae']
)

# ==========================================
# MODEL SUMMARY
# ==========================================

print("\nModel Summary")

model.summary()

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(

    X_train,
    Y_train,

    epochs=200,

    batch_size=32,
    validation_data=(X_test, Y_test),
    verbose=2
)

# ==========================================
# SAVE MODEL
# ==========================================

model.save(os.path.join(MODELS_DIR, "trajectory_lstm_model.h5"))

print("\nModel Saved Successfully")

# ==========================================
# PLOT TRAINING LOSS
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("LSTM Training Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()