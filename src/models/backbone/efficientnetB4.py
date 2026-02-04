import torch.nn as nn
import timm

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)