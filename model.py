import torch
import torch.nn as nn
from torchvision import models

class MultiTaskModel(nn.Module):
    """
    Multi-task model with separate classifiers for mask, glass, and cap predictions.
    """
    def __init__(self, base_model, feature_dim):
        super(MultiTaskModel, self).__init__()
        
        # Replace the classifier layer in the base model with Identity
        self.base_model = base_model
        self.base_model.classifier[1] = nn.Identity()

        # Define separate classifiers for each task
        self.classifier_mask = nn.Linear(feature_dim, 1)
        self.classifier_glass = nn.Linear(feature_dim, 1)
        self.classifier_cap = nn.Linear(feature_dim, 1)

    def forward(self, x):
        features = self.base_model(x)
        out_mask = self.classifier_mask(features)
        out_glass = self.classifier_glass(features)
        out_cap = self.classifier_cap(features)
        return out_mask, out_glass, out_cap


class ModelFactory:
    """
    Factory for creating models based on configuration.
    Supports single-task and multi-task models with dynamic feature dimensions.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def get_feature_dim(self, base_model):
        """
        Dynamically fetch the feature dimension from the base model.
        """
        if self.cfg.type == "mbn":
            return base_model.last_channel
        elif self.cfg.type.startswith('resnet'):
            return base_model.fc.in_features
        elif self.cfg.type == "efficientnet_b0":
            return base_model.classifier[1].in_features
        elif self.cfg.type == "vgg16":
            return base_model.classifier[6].in_features
        else:
            raise ValueError(f"Unsupported model type: {self.cfg.type}")

    def create_base_model(self):
        """
        Create the base model according to the model type in the configuration.
        """
        if self.cfg.type == "mbn":
            base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif self.cfg.type == "resnet18":
            base_model = models.resnet18(pretrained=True)
        elif self.cfg.type == "resnet50":
            base_model = models.resnet50(pretrained=True)
        elif self.cfg.type == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif self.cfg.type == "vgg16":
            base_model = models.vgg16(pretrained=True)
        else:
            raise ValueError(f"Unsupported model type: {self.cfg.type}")
        return base_model

    def initialize_model(self):
        """
        Initialize the model based on the configuration:
        - Single-task: Modify the final classifier for num_output classes.
        - Multi-task: Use MultiTaskModel for task-specific outputs.
        """
        base_model = self.create_base_model()

        feature_dim = self.get_feature_dim(base_model)
        if self.cfg.multitask:
            model = MultiTaskModel(base_model, feature_dim)
        else:
            if self.cfg.type in ["mbn", "efficientnet_b0"]:
                base_model.classifier[1] = nn.Linear(feature_dim, self.cfg.num_output)
            elif self.cfg.type.startswith('resnet'):
                base_model.fc = nn.Linear(feature_dim, self.cfg.num_output)
            elif self.cfg.type == "vgg16":
                base_model.classifier[6] = nn.Linear(feature_dim, self.cfg.num_output)
            model = base_model

        return model
    @staticmethod
    def build_model(cfg):
        """
        Static method to initialize and return a model.
        """
        return ModelFactory(cfg).initialize_model()