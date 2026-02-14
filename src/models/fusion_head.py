import torch
import torch.nn as nn
from typing import Optional, Dict
from .backbone.efficientnetB4 import EfficientNetBackbone
from .backbone.convnextLarge import ConvNeXtBackbone # Đảm bảo file này tồn tại
from .backbone.resnet50 import ResNet50Backbone

class MultimodalClassifier(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, cat_cardinalities: Optional[Dict[str, int]] = None,
                 num_numeric=0, emb_dim=8, num_classes=1, use_metadata=True, meta_weight=1.0):
        super().__init__()
        self.use_metadata = use_metadata
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or {}
        self.emb_dim = emb_dim
        self.meta_weight = meta_weight

        # --- LOGIC CHỌN BACKBONE ---
        if self.backbone_type == 'convnext':
            self.backbone = ConvNeXtBackbone(
                model_name='convnext_large',
                pretrained=pretrained
            )

        elif self.backbone_type == 'efficientnet':
            self.backbone = EfficientNetBackbone(
                model_name='tf_efficientnet_b4_ns',
                pretrained=pretrained
            )

        elif self.backbone_type == 'resnet':
            self.backbone = ResNet50Backbone(
                model_name='resnet50',
                pretrained=pretrained
            )

        else:
            raise ValueError(f"Backbone không hợp lệ: {self.backbone_type}")

        # Lấy số chiều feature
        self.img_features_dim = self.backbone.num_features

    def forward(self, x_img, meta_num=None, meta_cat=None):
        if x_img.dtype == torch.float16: x_img = x_img.float()
        feat_img = self.backbone(x_img)

        if self.use_metadata and (meta_num is not None and meta_cat is not None):
            emb_list = [self.emb_layers[c](meta_cat[:, i]) for i, c in enumerate(self.cat_names)]
            emb_concat = torch.cat(emb_list, dim=1) if emb_list else torch.zeros((feat_img.size(0), 0),
                                                                                 device=feat_img.device)

            meta_input = torch.cat([meta_num, emb_concat], dim=1) if self.num_numeric > 0 else emb_concat
            
            if meta_input.dtype == torch.float16: meta_input = meta_input.float()

            feat_meta = self.metadata_mlp(meta_input) * self.meta_weight
            feat = torch.cat([feat_img, feat_meta], dim=1)
        else:
            feat = feat_img

        return self.classifier(feat)

class DualEmbeddingFusion(nn.Module):
    def __init__(self, backbone_type='efficientnet', pretrained=True, 
                 cat_cardinalities=None, num_numeric=0, num_classes=1, embed_dim=256):
        super().__init__()
        
        # Chọn backbone tương tự như class MultimodalModel
        if backbone_type == 'convnext':
            self.backbone = ConvNeXtBackbone(model_name='convnext_large', pretrained=pretrained)
        else:
            self.backbone = EfficientNetBackbone('tf_efficientnet_b4_ns', pretrained=pretrained)
            
        self.img_dim = self.backbone.num_features 
        self.meta_dim = num_numeric + len(cat_cardinalities) if cat_cardinalities else num_numeric
        
        self.meta_embed = nn.Sequential(nn.Linear(self.meta_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim))
        self.img_embed = nn.Linear(self.img_dim, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x_img, meta_vec):
        img_feat = self.backbone(x_img)
        img_emb = self.img_embed(img_feat)
        meta_emb = self.meta_embed(meta_vec)
        gate_val = torch.sigmoid(self.gate(torch.cat([img_emb, meta_emb], dim=1)))
        fusion = gate_val * img_emb + (1 - gate_val) * meta_emb
        return self.classifier(fusion)