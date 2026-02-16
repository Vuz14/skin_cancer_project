import torch
import torch.nn as nn
from typing import Optional, Dict
from .backbone.efficientnetB4 import EfficientNetBackbone
from .backbone.resnet50 import ResNet50Backbone

import torch
import torch.nn as nn
from typing import Optional, Dict
from .backbone.efficientnetB4 import EfficientNetBackbone
from .backbone.resnet50 import ResNet50Backbone


class MultimodalClassifier(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True, cat_cardinalities: Optional[Dict[str, int]] = None,
                 num_numeric=0, emb_dim=8, num_classes=1, use_metadata=True, meta_weight=1.0):
        super().__init__()
        self.use_metadata = use_metadata
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or {}
        self.emb_dim = emb_dim
        self.meta_weight = meta_weight

        # --- LOGIC CHỌN BACKBONE ---
        if 'resnet' in model_name.lower():
            self.backbone = ResNet50Backbone(model_name, pretrained)
        else:
            self.backbone = EfficientNetBackbone(model_name, pretrained)

        self.img_features_dim = self.backbone.num_features

        # --- Phần Embedding & Classifier giữ nguyên như cũ ---
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
            # FiLM generator: generates gamma and beta for modulation
            self.film_generator = nn.Sequential(
                nn.Linear(input_meta_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, self.img_features_dim * 2)
            )
            # Initialize FiLM parameters to zero (identity transformation initially)
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
            self.metadata_mlp = None  # Not used with FiLM
        else:
            self.film_generator = None
            self.metadata_mlp = None
            self.classifier = nn.Sequential(
                nn.Linear(self.img_features_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
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

            # FiLM: Generate gamma and beta from metadata
            film_params = self.film_generator(meta_input)
            gamma, beta = torch.split(film_params, self.img_features_dim, dim=1)
            
            # Apply FiLM modulation: (1 + gamma) * feat + beta
            feat = (1 + gamma) * feat_img + beta
        else:
            feat = feat_img

        return self.classifier(feat)

class DualEmbeddingFusion(nn.Module):
    def __init__(
        self,
        pretrained=True,
        cat_cardinalities=None,
        num_numeric=0,
        num_classes=1,
        embed_dim=256,
        meta_weight=1.0,
        emb_dim=8
    ):
        super().__init__()

        self.meta_weight = meta_weight
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or {}
        self.cat_names = list(self.cat_cardinalities.keys())
        self.emb_dim = emb_dim

        # ===== IMAGE BACKBONE =====
        self.backbone = EfficientNetBackbone('tf_efficientnet_b4_ns', pretrained)
        self.img_dim = self.backbone.num_features

        # ===== CATEGORICAL EMBEDDINGS =====
        self.emb_layers = nn.ModuleDict()
        total_emb_dim = 0
        for cname in self.cat_names:
            card = int(self.cat_cardinalities[cname])
            d = min(self.emb_dim, max(2, int(card ** 0.5)))
            self.emb_layers[cname] = nn.Embedding(card, d)
            total_emb_dim += d

        meta_input_dim = total_emb_dim + num_numeric

        # ===== METADATA MLP =====
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # ===== IMAGE EMBEDDING =====
        self.img_embed = nn.Sequential(
            nn.Linear(self.img_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        # ===== GATE =====
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # ===== CLASSIFIER =====
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, meta_num, meta_cat):
        feat_img = self.backbone(x_img)

        # ----- categorical embedding -----
        emb_list = [self.emb_layers[c](meta_cat[:, i]) for i, c in enumerate(self.cat_names)]
        emb_concat = torch.cat(emb_list, dim=1) if emb_list else torch.zeros(
            (feat_img.size(0), 0), device=feat_img.device
        )

        meta_input = torch.cat([meta_num, emb_concat], dim=1) if self.num_numeric > 0 else emb_concat

        feat_meta = self.meta_mlp(meta_input)

        # ===== clamp metadata weight =====
        feat_meta = feat_meta * min(self.meta_weight, 2.0)

        img_emb = self.img_embed(feat_img)

        # ===== normalize before fusion =====
        img_emb = nn.functional.normalize(img_emb, dim=1)
        feat_meta = nn.functional.normalize(feat_meta, dim=1)

        # ===== gated fusion =====
        gate_val = torch.sigmoid(self.gate(torch.cat([img_emb, feat_meta], dim=1)))
        fusion = gate_val * img_emb + (1 - gate_val) * feat_meta

        return self.classifier(fusion)
