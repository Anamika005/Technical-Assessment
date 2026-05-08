import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Base directory = d:\Assesment  (two levels up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "processed")

# ==========================================
# LOAD CSV
# ==========================================

# NOTE: actual filename on disk has a typo — 'trajcetories' (missing 'e')
DATA_FILE = os.path.join(PROCESSED_DIR, "trajcetories.csv")

df = pd.read_csv(DATA_FILE)

print("\nOriginal Dataset:")
print(df.head())

# ==========================================
# SELECT FEATURES
# ==========================================

features = df[[
    "current_x",
    "current_y"
]]

targets = df[[
    "future_x",
    "future_y"
]]

# ==========================================
# NORMALIZATION
# ==========================================

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_scaled = feature_scaler.fit_transform(features)
targets_scaled = target_scaler.fit_transform(targets)

# ==========================================
# CREATE SEQUENCES
# ==========================================

SEQUENCE_LENGTH = 10

X = []
Y = []

for i in range(len(features_scaled) - SEQUENCE_LENGTH):

    X.append(
        features_scaled[i:i + SEQUENCE_LENGTH]
    )

    Y.append(
        targets_scaled[i + SEQUENCE_LENGTH]
    )

X = np.array(X)
Y = np.array(Y)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# PRINT SHAPES
# ==========================================

print("\nTraining Shapes")

print("X_train Shape:", X_train.shape)
print("Y_train Shape:", Y_train.shape)

print("\nTesting Shapes")

print("X_test Shape:", X_test.shape)
print("Y_test Shape:", Y_test.shape)

# ==========================================
# SAVE PREPROCESSED DATA
# ==========================================
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)

np.save(os.path.join(PROCESSED_DIR, "Y_train.npy"), Y_train)
np.save(os.path.join(PROCESSED_DIR, "Y_test.npy"), Y_test)

print("\nPreprocessed Data Saved Successfully")