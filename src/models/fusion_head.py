import torch
import torch.nn as nn
from typing import Optional, Dict
from .backbone.convnext import ConvNeXtBackbone
from .backbone.efficientnetB4 import EfficientNetBackbone
from .backbone.resnet50 import ResNet50Backbone
from .backbone.vit16 import ViT16Backbone


def build_image_backbone(model_name: str, pretrained: bool):
    model_name_lower = model_name.lower()
    if "convnext" in model_name_lower:
        return ConvNeXtBackbone(model_name, pretrained)
    if "vit" in model_name_lower:
        return ViT16Backbone(model_name, pretrained)
    if "resnet" in model_name_lower:
        return ResNet50Backbone(model_name, pretrained)
    return EfficientNetBackbone(model_name, pretrained)

class MultimodalClassifier(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True, cat_cardinalities: Optional[Dict[str, int]] = None,
                 num_numeric=0, emb_dim=8, num_classes=1, use_metadata=True, meta_weight=1.0):
        super().__init__()
        self.use_metadata = use_metadata
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or {}
        self.emb_dim = emb_dim
        self.meta_weight = meta_weight # Trá»ng sá»‘ Ä‘iá»u chá»‰nh má»©c Ä‘á»™ áº£nh hÆ°á»Ÿng cá»§a FiLM

        # --- CHá»ŒN BACKBONE ---
        self.backbone = build_image_backbone(model_name, pretrained)
        self.img_features_dim = self.backbone.num_features

        # --- Cáº¤U HÃŒNH METADATA ---
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
            # FiLM generator: Táº¡o gamma vÃ  beta
            self.film_generator = nn.Sequential(
                nn.Linear(input_meta_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.img_features_dim * 2)
            )
            # Khá»Ÿi táº¡o vá» 0 Ä‘á»ƒ ban Ä‘áº§u mÃ´ hÃ¬nh coi nhÆ° chÆ°a cÃ³ metadata
            self.film_generator[-1].weight.data.zero_()
            self.film_generator[-1].bias.data.zero_()
            
            self.classifier = nn.Sequential(
                nn.Linear(self.img_features_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )
        else:
            self.film_generator = None
            self.classifier = nn.Sequential(
                nn.Linear(self.img_features_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    def forward(self, x_img, meta_num=None, meta_cat=None):
        if x_img.dtype == torch.float16: x_img = x_img.float()
        feat_img = self.backbone(x_img)

        # Check nelement chuáº©n nhÆ° code cÅ© cá»§a báº¡n
        has_meta_input = (meta_num is not None and meta_num.nelement() > 0) and \
                         (meta_cat is not None and meta_cat.nelement() > 0)

        if self.use_metadata and has_meta_input:
            # 1. Xá»­ lÃ½ Categorical
            emb_list = [self.emb_layers[c](meta_cat[:, i]) for i, c in enumerate(self.cat_names)]
            emb_concat = torch.cat(emb_list, dim=1)

            # 2. Káº¿t há»£p vá»›i Numeric
            meta_input = torch.cat([meta_num, emb_concat], dim=1) if self.num_numeric > 0 else emb_concat
            if meta_input.dtype == torch.float16: meta_input = meta_input.float()

            # 3. FiLM modulation: (1 + gamma * weight) * feat + (beta * weight)
            film_params = self.film_generator(meta_input)
            gamma, beta = torch.split(film_params, self.img_features_dim, dim=1)
            
            # Cáº£i tiáº¿n: ThÃªm meta_weight Ä‘á»ƒ Ä‘iá»u tiáº¿t sá»©c máº¡nh cá»§a metadata
            feat = (1 + gamma * self.meta_weight) * feat_img + (beta * self.meta_weight)
        else:
            feat = feat_img

        return self.classifier(feat)

class DualEmbeddingFusion(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True, cat_cardinalities=None,
                 num_numeric=0, num_classes=1, embed_dim=256, meta_weight=1.0, emb_dim=8, use_metadata=True):
        super().__init__()
        self.use_metadata = use_metadata
        self.meta_weight = meta_weight
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or {}
        self.cat_names = list(self.cat_cardinalities.keys())
        self.emb_dim = emb_dim

        self.backbone = build_image_backbone(model_name, pretrained)
        self.img_dim = self.backbone.num_features

        self.emb_layers = nn.ModuleDict()
        total_emb_dim = 0
        for cname in self.cat_names:
            card = int(self.cat_cardinalities[cname])
            d = min(self.emb_dim, max(2, int(card ** 0.5)))
            self.emb_layers[cname] = nn.Embedding(card, d)
            total_emb_dim += d

        meta_input_dim = total_emb_dim + num_numeric

        if self.use_metadata and meta_input_dim > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
            self.img_embed = nn.Linear(self.img_dim, embed_dim)
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1)
            )
            final_in_dim = embed_dim
        else:
            self.use_metadata = False
            final_in_dim = self.img_dim

        self.classifier = nn.Sequential(
            nn.Linear(final_in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, meta_num=None, meta_cat=None):
        feat_img = self.backbone(x_img)

        has_meta = (meta_num is not None and meta_num.nelement() > 0) and \
                   (meta_cat is not None and meta_cat.nelement() > 0)

        if self.use_metadata and has_meta:
            # FIX Lá»–I "feat_meta is not defined":
            emb_list = [self.emb_layers[c](meta_cat[:, i]) for i, c in enumerate(self.cat_names)]
            emb_concat = torch.cat(emb_list, dim=1)
            meta_input = torch.cat([meta_num, emb_concat], dim=1) if self.num_numeric > 0 else emb_concat
            
            # TÃ­nh toÃ¡n feat_meta tá»« dá»¯ liá»‡u Ä‘áº§u vÃ o
            feat_meta = self.meta_mlp(meta_input.float())

            feat_img_proj = self.img_embed(feat_img)
            gate_input = torch.cat([feat_img_proj, feat_meta], dim=1)
            gate_score = torch.sigmoid(self.gate(gate_input))
            
            # Gating Fusion
            feat = gate_score * feat_img_proj + (1 - gate_score) * feat_meta
        else:
            feat = self.img_embed(feat_img) if self.use_metadata else feat_img

        return self.classifier(feat)


class ConcatenationFusion(nn.Module):
    """Direct image/clinical concatenation baseline (Strategy 2)."""

    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True, cat_cardinalities=None,
                 num_numeric=0, num_classes=1, emb_dim=8, use_metadata=True):
        super().__init__()
        self.use_metadata = use_metadata
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or {}
        self.cat_names = list(self.cat_cardinalities.keys())

        self.backbone = build_image_backbone(model_name, pretrained)
        self.img_dim = self.backbone.num_features

        self.emb_layers = nn.ModuleDict()
        total_emb_dim = 0
        for cname in self.cat_names:
            card = int(self.cat_cardinalities[cname])
            dim = min(emb_dim, max(2, int(card ** 0.5)))
            self.emb_layers[cname] = nn.Embedding(card, dim)
            total_emb_dim += dim

        meta_dim = total_emb_dim + max(0, num_numeric)
        if use_metadata and meta_dim > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            classifier_dim = self.img_dim + 128
        else:
            self.use_metadata = False
            self.meta_mlp = None
            classifier_dim = self.img_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_img, meta_num=None, meta_cat=None):
        feat_img = self.backbone(x_img.float())
        has_meta = (meta_num is not None and meta_num.nelement() > 0) and \
                   (meta_cat is not None and meta_cat.nelement() > 0)

        if self.use_metadata and has_meta:
            embeddings = [self.emb_layers[name](meta_cat[:, i]) for i, name in enumerate(self.cat_names)]
            meta_input = torch.cat([meta_num] + embeddings, dim=1) if self.num_numeric > 0 else torch.cat(embeddings, dim=1)
            feat = torch.cat([feat_img, self.meta_mlp(meta_input.float())], dim=1)
        else:
            feat = feat_img

        return self.classifier(feat)
