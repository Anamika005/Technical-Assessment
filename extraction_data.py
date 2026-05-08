import os
import tensorflow as tf
import pandas as pd
import numpy as np

# Base directory = d:\Assesment  (two levels up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "Data", "Raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "processed")

# ==========================================
# FILE PATH
# ==========================================

TFRECORD_FILE = os.path.join(RAW_DIR, "training_tfexample.tfrecord-00000-of-01000")

# ==========================================
# LOAD DATASET
# ==========================================

dataset = tf.data.TFRecordDataset(TFRECORD_FILE)

# ==========================================
# FEATURES TO EXTRACT
# ==========================================

feature_description = {
    'state/current/x': tf.io.VarLenFeature(tf.float32),
    'state/current/y': tf.io.VarLenFeature(tf.float32),
    'state/future/x': tf.io.VarLenFeature(tf.float32),
    'state/future/y': tf.io.VarLenFeature(tf.float32),
}

# ==========================================
# PARSE FUNCTION
# ==========================================

def parse_example(example_proto):

    return tf.io.parse_single_example(
        example_proto,
        feature_description
    )

# ==========================================
# EXTRACT DATA
# ==========================================

data = []

MAX_SAMPLES = 200

for i, raw_record in enumerate(dataset):

    if i >= MAX_SAMPLES:
        break

    example = parse_example(raw_record)

    current_x = tf.sparse.to_dense(
        example['state/current/x']
    ).numpy()

    current_y = tf.sparse.to_dense(
        example['state/current/y']
    ).numpy()

    future_x = tf.sparse.to_dense(
        example['state/future/x']
    ).numpy()

    future_y = tf.sparse.to_dense(
        example['state/future/y']
    ).numpy()

    # ======================================
    # STORE TRAJECTORY POINTS
    # ======================================

    sequence_length = min(
        len(current_x),
        len(future_x),
        10
    )

    for j in range(sequence_length):

        row = {

            "sample_id": i,

            "current_x": current_x[j],
            "current_y": current_y[j],

            "future_x": future_x[j],
            "future_y": future_y[j]
        }

        data.append(row)

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(data)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# SAVE CSV
# ==========================================

# NOTE: saving with the same filename that exists on disk (typo intentional)
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "trajcetories.csv")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTrajectory CSV Saved Successfully")
print(f"Saved at: {OUTPUT_FILE}")