import numpy as np
import cv2

def preprocess_camera_frame(image_array):

    array = image_array

    array = array[:, :, :3]

    array = array[:, :, ::-1]

    resized = cv2.resize(array, (224, 224))

    resized = resized.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (resized - mean) / std

    processed_image = np.transpose(normalized, (2, 0, 1))

    return processed_image

def preprocess_lidar_point_cloud(carla_lidar_data):

    points = np.frombuffer(carla_lidar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))

    mask = (
        (points[:, 0] >= -40) & (points[:, 0] <= 40) &
        (points[:, 1] >= -40) & (points[:, 1] <= 40) &
        (points[:, 2] >= -3.0) & (points[:, 2] <= 1.0)
    )
    roi_points = points[mask]

    return roi_points