import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# CAMERA PERCEPTION (ResNet-18)
# ==========================================

class CameraPerception:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"Initializing ResNet-18 on {self.device}...")
        
        # Load Pretrained ResNet-18 (ImageNet)
        self.model = models.resnet18(pretrained=True)
        
        # We drop the final classification layer to use it as a feature extractor
        # It outputs a 512-dimensional feature vector
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        print("ResNet-18 Initialized Successfully")

    def extract_features(self, preprocessed_image):
        """
        Extracts a 512-dimensional feature vector from the preprocessed camera image.
        Expected input shape: (3, 224, 224) as numpy array.
        """
        # Convert to tensor, cast to float32, and add batch dimension: (1, 3, 224, 224)
        image_tensor = torch.from_numpy(preprocessed_image).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Output shape: (1, 512, 1, 1)
            features = self.feature_extractor(image_tensor)
            
            # Flatten to (512,)
            features = features.squeeze().cpu().numpy()
            
        return features

# ==========================================
# LIDAR PERCEPTION (PointPillars)
# ==========================================

class LidarPerception:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"Initializing PointPillars on {self.device}...")
        
        # NOTE: Full PointPillars implementation typically relies on OpenPCDet or mmdet3d.
        # This acts as a wrapper class. For a full integration, you would load the 
        # pre-trained Waymo checkpoint here.
        # Example: self.model = build_network(model_cfg, num_class=3)
        #          self.model.load_params_from_file('pointpillars_waymo.pth')
        
        self.initialized = True
        print("PointPillars Wrapper Initialized")

    def detect_objects(self, preprocessed_points):
        """
        Runs the PointPillars model to detect 3D Bounding Boxes.
        Outputs a list of detected objects (e.g., pedestrians, vehicles).
        """
        # Placeholder for actual inference logic
        # detections = self.model(preprocessed_points)
        # return detections
        
        # Simulated Detections for Pipeline testing
        simulated_detections = []
        return simulated_detections
