import os
import subprocess
import sys
import time

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTION_SCRIPT = os.path.join(SOURCE_DIR, "extraction_data.py")
PREPROCESS_SCRIPT = os.path.join(SOURCE_DIR, "preprocess.py")
TRAIN_SCRIPT      = os.path.join(SOURCE_DIR, "train_code.py")

def run_script(script_path, name):

    print(f"\n{'='*60}")
    print(f" STEP: {name}")
    print(f" Running: {os.path.basename(script_path)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:

        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] {name} completed in {elapsed:.2f} seconds.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {name} failed with exit code {e.returncode}.")
        return False
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        return False

def main():
    print("Starting Trajectory Prediction Pipeline...")

    if not run_script(EXTRACTION_SCRIPT, "DATA EXTRACTION"):
        print("Pipeline aborted at Extraction stage.")
        return

    if not run_script(PREPROCESS_SCRIPT, "DATA PREPROCESSING"):
        print("Pipeline aborted at Preprocessing stage.")
        return

    if not run_script(TRAIN_SCRIPT, "MODEL TRAINING"):
        print("Pipeline aborted at Training stage.")
        return

    print(f"\n{'='*60}")
    print(" PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print("Final model is saved in: Data/models/trajectory_lstm_model.h5")

if __name__ == "__main__":
    main()