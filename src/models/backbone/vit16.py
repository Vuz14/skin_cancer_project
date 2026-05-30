import torch
import torch.nn as nn
import timm

class ViT16Backbone(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # timm vit thường có num_features hoặc embed_dim
        self.num_features = getattr(self.backbone, "num_features", None)
        if self.num_features is None:
            self.num_features = getattr(self.backbone, "embed_dim", None)
        if self.num_features is None:
            raise AttributeError(
                f"Cannot infer num_features for model '{model_name}'. "
                "Expected timm model to have 'num_features' or 'embed_dim'."
            )

    def forward(self, x):
        # Trainer có thể chạy AMP -> tránh float16 issue giống EfficientNetB4_Multimodal
        if x.dtype == torch.float16:
            x = x.float()
        return self.backbone(x)
