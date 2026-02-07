import torch.nn as nn
import timm

class ResNet50Backbone(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        # num_classes=0 sẽ loại bỏ lớp classification head (fc layer)
        # và trả về feature vector (thường là sau lớp pooling cuối cùng)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)