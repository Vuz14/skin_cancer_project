import torch
import torch.nn as nn
import timm


class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name: str = "convnext_base", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.num_features = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        return self.backbone(x)
