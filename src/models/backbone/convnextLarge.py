import timm
import torch.nn as nn

class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name="convnext_large", pretrained=True):
        super().__init__()

        # ✅ tạo backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,   # remove classifier
            global_pool="avg"
        )

        # ✅ số feature output
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)