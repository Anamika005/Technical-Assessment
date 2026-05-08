import numpy as np
import cv2

# ==========================================
# CAMERA PREPROCESSING (ResNet-18)
# ==========================================

def preprocess_camera_frame(image_array):
    """
    Preprocess a CARLA RGB camera frame for ResNet-18.
    ResNet-18 expects 224x224 RGB images, normalized.
    """
    # image_array is already a numpy array of shape (H, W, 4) from the callback
    array = image_array
    
    # Extract RGB and drop Alpha
    array = array[:, :, :3]
    
    # Convert BGR to RGB
    array = array[:, :, ::-1]
    
    # Resize to 224x224 for ResNet
    resized = cv2.resize(array, (224, 224))
    
    # Normalize (ImageNet standards)
    resized = resized.astype(np.float32) / 255.0
    
    # mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (resized - mean) / std
    
    # Transpose to Channels First (C, H, W) for PyTorch
    processed_image = np.transpose(normalized, (2, 0, 1))
    
    return processed_image

# ==========================================
# LIDAR PREPROCESSING (PointPillars)
# ==========================================

def preprocess_lidar_point_cloud(carla_lidar_data):
    """
    Convert raw CARLA LiDAR point cloud into a format suitable for PointPillars.
    PointPillars typically requires voxelization/pillarization.
    """
    # Convert raw bytes to numpy array of shape (N, 4) -> (X, Y, Z, Intensity)
    points = np.frombuffer(carla_lidar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    
    # Filter points based on Region of Interest (ROI)
    # Example: X in [-40, 40], Y in [-40, 40], Z in [-3, 1]
    mask = (
        (points[:, 0] >= -40) & (points[:, 0] <= 40) &
        (points[:, 1] >= -40) & (points[:, 1] <= 40) &
        (points[:, 2] >= -3.0) & (points[:, 2] <= 1.0)
    )
    roi_points = points[mask]
    
    # Note: Full PointPillars voxelization requires creating pillars, computing 
    # pillar centers, and padding. This is typically handled by the PointPillars 
    # model data loader. Here we return the filtered (N, 4) array.
    return roi_points
