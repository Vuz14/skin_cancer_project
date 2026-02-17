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

        # --- CHỌN BACKBONE ---
        if 'resnet' in model_name.lower():
            self.backbone = ResNet50Backbone(model_name, pretrained)
        else:
            self.backbone = EfficientNetBackbone(model_name, pretrained)

        self.img_features_dim = self.backbone.num_features

        # --- CẤU HÌNH METADATA ---
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

        # Kiểm tra tính hợp lệ của metadata đầu vào
        has_meta_input = (meta_num is not None and meta_num.nelement() > 0) and \
                         (meta_cat is not None and meta_cat.nelement() > 0)

        if self.use_metadata and has_meta_input:
            # 1. Xử lý Categorical
            emb_list = [self.emb_layers[c](meta_cat[:, i]) for i, c in enumerate(self.cat_names)]
            emb_concat = torch.cat(emb_list, dim=1) if emb_list else torch.zeros((feat_img.size(0), 0), device=feat_img.device)

            # 2. Kết hợp với Numeric
            meta_input = torch.cat([meta_num, emb_concat], dim=1) if self.num_numeric > 0 else emb_concat
            if meta_input.dtype == torch.float16: meta_input = meta_input.float()

            # FiLM: Generate gamma and beta from metadata
            film_params = self.film_generator(meta_input)
            gamma, beta = torch.split(film_params, self.img_features_dim, dim=1)
            
            # Apply FiLM modulation: (1 + gamma) * feat + beta
            feat = (1 + gamma) * feat_img + beta
        else:
            # Mode diag1: Bỏ qua hoàn toàn metadata
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

        # Backbone
        if 'resnet' in model_name.lower():
            self.backbone = ResNet50Backbone(model_name, pretrained)
        else:
            self.backbone = EfficientNetBackbone(model_name, pretrained)
        self.img_dim = self.backbone.num_features

        # Embedding logic
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

        # Kiểm tra nelement() để biết tensor có rỗng hay không
        has_meta = (meta_num is not None and meta_num.nelement() > 0)

        if self.use_metadata and has_meta:
            # Xử lý metadata như bình thường...
            # feat_meta = self.metadata_mlp(...)
            feat = torch.cat([feat_img, feat_meta], dim=1)
        else:
            # Ngắt metadata hoàn toàn
            feat = feat_img

        return self.classifier(feat)