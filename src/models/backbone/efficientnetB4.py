import torch
import torch.nn as nn
import timm

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True):
        super().__init__()
        # global_pool='' để lấy feature map 2D phục vụ Grad-CAM tốt hơn
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        self.num_features = self.model.num_features 

    def forward(self, x):
        features_2d = self.model(x) 

        pooled_features = torch.mean(features_2d, dim=[2, 3]) 
        
        return pooled_features