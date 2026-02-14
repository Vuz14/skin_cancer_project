import torch
import torch.nn as nn
from typing import Optional, Dict
from .backbone.vit16 import ViT16Backbone


class ViT16_Multimodal(nn.Module):
    def __init__(
        self,
        pretrained=True,
        cat_cardinalities: Optional[Dict[str, int]] = None,
        num_numeric=0,
        emb_dim=8,
        num_classes=1,
        use_metadata=True,
        meta_weight=1.0
    ):
        super().__init__()
        self.use_metadata = use_metadata
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or {}
        self.emb_dim = emb_dim
        self.meta_weight = meta_weight

        #  CHỈ ĐỔI BACKBONE
        self.backbone = ViT16Backbone(pretrained=pretrained)
        self.img_features_dim = self.backbone.num_features

        # ===== Embedding cho metadata categorical =====
        self.cat_names = list(self.cat_cardinalities.keys())
        self.emb_layers = nn.ModuleDict()
        total_emb_dim = 0
        for cname in self.cat_names:
            card = int(self.cat_cardinalities[cname])
            d = min(self.emb_dim, max(2, int(card ** 0.5)))
            self.emb_layers[cname] = nn.Embedding(card, d)
            total_emb_dim += d

        # ===== Metadata MLP + Classifier (GIỮ NGUYÊN LOGIC) =====
        if use_metadata:
            input_meta_dim = total_emb_dim + max(0, num_numeric)
            self.metadata_mlp = nn.Sequential(
                nn.Linear(input_meta_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            self.classifier = nn.Sequential(
                nn.Linear(self.img_features_dim + 32, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            self.metadata_mlp = None
            self.classifier = nn.Sequential(
                nn.Linear(self.img_features_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    def forward(self, x_img, meta_num=None, meta_cat=None):
        # Giữ tương thích AMP
        if x_img.dtype == torch.float16:
            x_img = x_img.float()

        feat_img = self.backbone(x_img)

        if self.use_metadata and (meta_num is not None and meta_cat is not None):
            emb_list = [
                self.emb_layers[c](meta_cat[:, i])
                for i, c in enumerate(self.cat_names)
            ]
            emb_concat = (
                torch.cat(emb_list, dim=1)
                if emb_list
                else torch.zeros((feat_img.size(0), 0), device=feat_img.device)
            )

            meta_input = (
                torch.cat([meta_num, emb_concat], dim=1)
                if self.num_numeric > 0
                else emb_concat
            )

            if meta_input.dtype == torch.float16:
                meta_input = meta_input.float()

            feat_meta = self.metadata_mlp(meta_input) * self.meta_weight
            feat = torch.cat([feat_img, feat_meta], dim=1)
        else:
            feat = feat_img

        return self.classifier(feat)
    
import torch
import torch.nn as nn
import timm

class ViTBackbone(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features  # 768

    def forward(self, x):
        return self.backbone(x)  # (B, 768)

class ViT16_DualEmbeddingFusion(nn.Module):
    """
    LATE FUSION đúng với trainer.py của repo:
      forward(x_img, meta_vec)
    """
    def __init__(self, pretrained=True, meta_dim=48, num_classes=1, embed_dim=256):
        super().__init__()
        self.metadata_mode = "late_fusion"

        self.backbone = ViTBackbone("vit_base_patch16_224", pretrained=pretrained)
        self.img_dim = self.backbone.num_features  # 768

        self.meta_dim = int(meta_dim)
        self.embed_dim = int(embed_dim)

        self.meta_embed = nn.Sequential(
            nn.Linear(self.meta_dim, 128), nn.ReLU(),
            nn.Linear(128, self.embed_dim)
        )
        self.img_embed = nn.Linear(self.img_dim, self.embed_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, meta_vec):
        if x_img.dtype == torch.float16:
            x_img = x_img.float()
        img_feat = self.backbone(x_img)
        img_emb = self.img_embed(img_feat)

        if meta_vec.dtype == torch.float16:
            meta_vec = meta_vec.float()
        meta_emb = self.meta_embed(meta_vec)

        gate_val = torch.sigmoid(self.gate(torch.cat([img_emb, meta_emb], dim=1)))
        fusion = gate_val * img_emb + (1 - gate_val) * meta_emb
        return self.classifier(fusion)
