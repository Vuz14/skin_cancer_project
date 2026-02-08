import torch
import torch.nn as nn
from typing import Optional, Dict
from .backbone.efficientnetB4 import EfficientNetBackbone
from .backbone.resnet50 import ResNet50Backbone

class EfficientNetB4_Multimodal(nn.Module):
    def __init__(self, pretrained=True, cat_cardinalities: Optional[Dict[str, int]] = None, num_numeric=0, emb_dim=8,
                 num_classes=1, use_metadata=True, meta_weight=1.0):
        super().__init__()
        self.use_metadata = use_metadata
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or {}
        self.emb_dim = emb_dim
        self.meta_weight = meta_weight

        self.backbone = EfficientNetBackbone('tf_efficientnet_b4_ns', pretrained)
        self.img_features_dim = self.backbone.num_features

        self.cat_names = list(self.cat_cardinalities.keys())
        self.emb_layers = nn.ModuleDict()
        total_emb_dim = 0
        for cname in self.cat_names:
            card = int(self.cat_cardinalities[cname])
            d = min(self.emb_dim, max(2, int(card ** 0.5)))
            self.emb_layers[cname] = nn.Embedding(card, d)
            total_emb_dim += d

        if use_metadata:
            input_meta_dim = total_emb_dim + max(0, num_numeric)
            self.metadata_mlp = nn.Sequential(
                nn.Linear(input_meta_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(64, 32), nn.ReLU()
            )
            self.classifier = nn.Sequential(
                nn.Linear(self.img_features_dim + 32, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            self.metadata_mlp = None
            self.classifier = nn.Sequential(
                nn.Linear(self.img_features_dim, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    def forward(self, x_img, meta_num=None, meta_cat=None):
        if x_img.dtype == torch.float16: x_img = x_img.float()
        feat_img = self.backbone(x_img)
        if self.use_metadata and (meta_num is not None and meta_cat is not None):
            emb_list = [self.emb_layers[c](meta_cat[:, i]) for i, c in enumerate(self.cat_names)]
            emb_concat = torch.cat(emb_list, dim=1) if emb_list else torch.zeros((feat_img.size(0), 0), device=feat_img.device)
            meta_input = torch.cat([meta_num, emb_concat], dim=1) if self.num_numeric > 0 else emb_concat
            if meta_input.dtype == torch.float16: meta_input = meta_input.float()
            feat_meta = self.metadata_mlp(meta_input) * self.meta_weight
            feat = torch.cat([feat_img, feat_meta], dim=1)
        else:
            feat = feat_img
        return self.classifier(feat)

class DualEmbeddingFusion(nn.Module):
    def __init__(self, pretrained=True, cat_cardinalities=None, num_numeric=0, num_classes=1, embed_dim=256):
        super().__init__()
        self.backbone = EfficientNetBackbone('tf_efficientnet_b4_ns', pretrained)
        self.img_dim = self.backbone.num_features 
        self.meta_dim = num_numeric + len(cat_cardinalities)
        self.meta_embed = nn.Sequential(nn.Linear(self.meta_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim))
        self.img_embed = nn.Linear(self.img_dim, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x_img, meta_vec):
        img_feat = self.backbone(x_img)
        img_emb = self.img_embed(img_feat)
        meta_emb = self.meta_embed(meta_vec)
        gate_val = torch.sigmoid(self.gate(torch.cat([img_emb, meta_emb], dim=1)))
        fusion = gate_val * img_emb + (1 - gate_val) * meta_emb
        return self.classifier(fusion)


class ResNet50_Multimodal(nn.Module):
    def __init__(self, pretrained=True, cat_cardinalities=None, num_numeric=0, emb_dim=8,
                 num_classes=1, use_metadata=True, meta_weight=1.0):
        super().__init__()
        self.use_metadata = use_metadata
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or {}
        self.emb_dim = emb_dim
        self.meta_weight = meta_weight

        # --- THAY ĐỔI Ở ĐÂY: Dùng ResNet50Backbone ---
        self.backbone = ResNet50Backbone('resnet50', pretrained)
        self.img_features_dim = self.backbone.num_features  # Tự động lấy số channels (2048)

        # Phần xử lý Metadata giữ nguyên logic cũ
        self.cat_names = list(self.cat_cardinalities.keys())
        self.emb_layers = nn.ModuleDict()
        total_emb_dim = 0
        for cname in self.cat_names:
            card = int(self.cat_cardinalities[cname])
            d = min(self.emb_dim, max(2, int(card ** 0.5)))
            self.emb_layers[cname] = nn.Embedding(card, d)
            total_emb_dim += d

        if use_metadata:
            input_meta_dim = total_emb_dim + max(0, num_numeric)
            self.metadata_mlp = nn.Sequential(
                nn.Linear(input_meta_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(64, 32), nn.ReLU()
            )
            # Input của Classifier = img_features (2048) + meta_features (32)
            self.classifier = nn.Sequential(
                nn.Linear(self.img_features_dim + 32, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            self.metadata_mlp = None
            self.classifier = nn.Sequential(
                nn.Linear(self.img_features_dim, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

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