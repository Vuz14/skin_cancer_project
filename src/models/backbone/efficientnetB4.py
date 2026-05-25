import torch
import torch.nn as nn
import timm

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True):
        super().__init__()
        # This wrapper intentionally supports EfficientNet, ConvNeXt and ViT:
        # timm returns the globally pooled feature vector with num_classes=0.
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.model.num_features

    def forward(self, x):
        return self.model(x)
