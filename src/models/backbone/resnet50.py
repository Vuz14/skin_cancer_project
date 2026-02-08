import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(ResNet50Backbone, self).__init__()

        # Load pre-trained weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = resnet50(weights=weights)

        # Lưu số features output (ResNet50 là 2048)
        self.num_features = self.model.fc.in_features

        # Xóa lớp classification cuối cùng (fc) vì ta không dùng
        # Nhưng giữ lại các lớp conv để trích xuất đặc trưng
        del self.model.fc

    def forward_features(self, x):
        """
        Trả về Feature Map (Batch, 2048, H/32, W/32)
        Cần thiết cho Attention (CBAM) và LIME/GradCAM
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

    def forward(self, x):
        """
        Forward thông thường (nếu dùng độc lập không qua CBAM)
        """
        x = self.forward_features(x)
        # Global Average Pooling
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x