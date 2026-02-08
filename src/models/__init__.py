import torch
import torch.nn as nn
from typing import Optional, Dict

# Import module
from .backbone.efficientnetB4 import EfficientNetBackbone
from .backbone.resnet50 import ResNet50Backbone
from .fusion_head import FusionHead
from .attention import CBAM


class ResNetCBAM(nn.Module):
    """
    Wrapper class: Backbone -> CBAM -> Pooling -> FusionHead
    """

    def __init__(self, backbone, fusion_head, features_dim=2048):
        super(ResNetCBAM, self).__init__()
        self.backbone = backbone

        # Thêm module CBAM Attention
        self.cbam = CBAM(channels=features_dim)

        self.fusion = fusion_head

        # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x):
        # Trả về Feature Map 4D (cần thiết cho LIME/GradCAM)
        return self.backbone.forward_features(x)

    def forward(self, x, meta_num=None, meta_cat=None):
        # 1. Trích xuất đặc trưng không gian (Output: BxCxHxW)
        features = self.forward_features(x)

        # 2. Áp dụng Attention (Tinh chỉnh đặc trưng)
        features = self.cbam(features)

        # 3. Pooling & Flatten (Output: BxC)
        features = self.pool(features).flatten(1)

        # 4. Đưa vào Fusion Head
        return self.fusion(features, meta_num, meta_cat)


def get_model(config, cat_cardinalities, num_numeric):
    """
    Factory function khởi tạo model
    """
    model_name = config['MODEL_NAME']
    pretrained = config.get('PRETRAINED', True)

    # 1. Khởi tạo Backbone
    if model_name == 'resnet50':
        # ResNet50: Output channels = 2048
        # Lưu ý: Class ResNet50Backbone của bạn phải có hàm forward_features
        model_backbone = ResNet50Backbone('resnet50', pretrained)
        features_dim = 2048

    elif 'efficientnet' in model_name:
        # EfficientNet-B4: Output channels = 1792
        model_backbone = EfficientNetBackbone('tf_efficientnet_b4_ns', pretrained)
        features_dim = 1792

    else:
        raise ValueError(f"Model {model_name} chưa được hỗ trợ.")

    # 2. Khởi tạo Fusion Head
    meta_boost = config.get('METADATA_FEATURE_BOOST', 1.0)

    fusion_head = FusionHead(
        num_numeric=num_numeric,
        num_categorical=len(cat_cardinalities),
        cat_cardinalities=list(cat_cardinalities.values()),
        input_dim=features_dim,
        hidden_dim=512,
        output_dim=1,
        metadata_feature_boost=meta_boost
    )

    # 3. Kết hợp lại bằng Wrapper
    model = ResNetCBAM(model_backbone, fusion_head, features_dim=features_dim)

    return model