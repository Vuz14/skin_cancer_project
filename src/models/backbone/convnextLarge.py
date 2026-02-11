import torch
import torch.nn as nn
from torchvision import models

class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name='convnext_large', pretrained=True):
        super().__init__()
        # Khởi tạo model từ torchvision (giống code Colab của bạn)
        if model_name == 'convnext_large':
            weights = "IMAGENET1K_V1" if pretrained else None
            self.backbone = models.convnext_large(weights=weights)
        
        # Lấy số feature đầu ra (thường là 1536 với bản Large)
        self.num_features = self.backbone.classifier[2].in_features
        
        # Loại bỏ lớp classifier gốc để lấy raw features (Identity)
        self.backbone.classifier[2] = nn.Identity()

    def forward(self, x):
        return self.backbone(x)