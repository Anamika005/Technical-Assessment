import torch
import torch.nn as nn
import torchvision.models as models

class CameraPerception:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"Initializing ResNet-18 on {self.device}...")

        self.model = models.resnet18(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        print("ResNet-18 Initialized Successfully")

    def extract_features(self, preprocessed_image):

        image_tensor = torch.from_numpy(preprocessed_image).float().unsqueeze(0).to(self.device)

        with torch.no_grad():

            features = self.feature_extractor(image_tensor)

            features = features.squeeze().cpu().numpy()

        return features

class LidarPerception:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"Initializing PointPillars on {self.device}...")

        self.initialized = True
        print("PointPillars Wrapper Initialized")

    def detect_objects(self, preprocessed_points):

        simulated_detections = []
        return simulated_detections