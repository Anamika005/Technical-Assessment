# Autonomous Urban Driving with Safe RL and Multi-Sensor Perception

This repository contains a robust implementation of an autonomous driving system designed for urban environments using the CARLA simulator. The system integrates advanced perception, trajectory prediction, and a Safe Reinforcement Learning (Safe RL) agent with Lagrangian constraints.

## Project Architecture

The system is built with a modular architecture:

### 1. Perception Layer (`Source/perception.py`)
- **Camera Perception**: Uses a **ResNet-18** feature extractor (pre-trained on ImageNet) to process RGB camera feeds and generate 512-dimensional feature vectors.
- **LiDAR Perception**: A wrapper for 3D object detection (e.g., PointPillars) to identify pedestrians, cyclists, and other vehicles.

### 2. Prediction Layer (`Source/predict.py`)
- Implements an **LSTM-based Trajectory Predictor**.
- Predicts the future positions of surrounding obstacles (vehicles/pedestrians) based on their past movement history.
- Critical for proactive safety and collision avoidance.

### 3. Safe RL Agent (`Source/rl_agent.py`)
- **Primary Objective**: Optimizes for efficiency (speed), comfort (jerk reduction), and energy consumption.
- **Lagrangian Relaxation**: Implements a safety constraint where a multiplier (lambda) penalizes the policy when safety thresholds are violated.
- **Emergency Brake Layer (Safety Shield)**: A deterministic rule-based override that triggers immediate braking if the Time-To-Collision (TTC) falls below a critical threshold.

### 4. Simulation Integration (`Source/carla_integration.py`)
- A custom **Gymnasium-compatible environment** for CARLA.
- Supports multi-scenario testing:
    - **Weather**: Clear, Rain.
    - **Time of Day**: Day, Night.
    - **Obstacles**: Dynamic pedestrians and cyclists.

---

## Getting Started

### Prerequisites
- [CARLA Simulator](https://carla.org/) (Version 0.9.13+ recommended)
- Python 3.8+

### Installation
1. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training Pipeline
You can run the full training pipeline with a single command:
```bash
python Source/pipeline.py
```

Alternatively, you can run the steps individually:

1. **Data Extraction**: Extract trajectories from raw TFRecord data.
   ```bash
   python Source/extraction_data.py
   ```
2. **Preprocessing**: Normalize and sequence the data for the LSTM.
   ```bash
   python Source/preprocess.py
   ```
3. **Training**: Train the LSTM model.
   ```bash
   python Source/train_code.py
   ```

### Running the Simulation
1. Start your CARLA simulator.
2. Run the integration script to test the agent across various scenarios:
   ```bash
   python Source/carla_integration.py
   ```

## Key Features
- **Multi-Scenario Validation**: Tested in day, night, and rainy conditions.
- **Safety First**: Combines learned RL behaviors with a hard-coded safety shield for maximum reliability.
- **Trajectory Aware**: The agent doesn't just react to current positions but predicts future movements to avoid collisions before they occur.

## Deliverables
- **Complete Source Code**: Modular Python implementation.
- **Training Setup**: Full pipeline from raw data to saved model.
- **Documentation**: Detailed overview of each component.
