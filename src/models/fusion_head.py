import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, num_numeric, num_categorical, cat_cardinalities, input_dim, hidden_dim=512, output_dim=1,
                 metadata_feature_boost=1.0):
        """
        Fusion Head: Module chuyên trách kết hợp đặc trưng hình ảnh và Metadata.
        """
        super(FusionHead, self).__init__()

        # 1. Xử lý Metadata (Categorical + Numeric)
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.metadata_feature_boost = metadata_feature_boost

        # Embedding cho Categorical Features
        # Tạo danh sách các lớp Embedding dựa trên số lượng giá trị (cardinality) của từng biến
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, min(50, (cardinality + 1) // 2))
            for cardinality in cat_cardinalities
        ])

        # Tính tổng kích thước vector sau khi embed
        self.emb_out_dim = sum([emb.embedding_dim for emb in self.embeddings])

        # MLP cho Metadata (Numeric + Categorical)
        self.meta_in_dim = self.emb_out_dim + num_numeric
        self.meta_out_dim = 64  # Kích thước vector metadata mong muốn

        if self.meta_in_dim > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(self.meta_in_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, self.meta_out_dim),
                nn.BatchNorm1d(self.meta_out_dim),
                nn.ReLU()
            )
        else:
            self.meta_mlp = None
            self.meta_out_dim = 0

        # 2. Fusion Layer (Image Features + Metadata Features)
        # Input dim = Image Features (từ Backbone) + Metadata Features (từ MLP)
        self.final_in_dim = input_dim + self.meta_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.final_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, img_features, meta_num=None, meta_cat=None):
        """
        Args:
            img_features: (Batch, Input_Dim) - Vector đặc trưng ảnh đã qua CBAM và Pooling
            meta_num: (Batch, Num_Numeric) - Metadata dạng số
            meta_cat: (Batch, Num_Categorical) - Metadata dạng phân loại
        """

        # --- NHÁNH METADATA ---
        meta_repr = []

        # 1. Embed categorical
        if self.num_categorical > 0 and meta_cat is not None:
            for i, emb in enumerate(self.embeddings):
                # Clamp để đảm bảo index không vượt quá giới hạn (an toàn)
                val = meta_cat[:, i].clamp(0, emb.num_embeddings - 1)
                meta_repr.append(emb(val))

        # 2. Nối với numeric
        if self.num_numeric > 0 and meta_num is not None:
            meta_repr.append(meta_num)

        # 3. Qua MLP để tạo vector đặc trưng metadata
        if len(meta_repr) > 0 and self.meta_mlp is not None:
            meta_vec = torch.cat(meta_repr, dim=1)
            meta_features = self.meta_mlp(meta_vec)
            # Boost metadata (tăng cường tín hiệu metadata nếu cần)
            meta_features = meta_features * self.metadata_feature_boost
        else:
            meta_features = None

        # --- FUSION & CLASSIFY ---
        if meta_features is not None:
            # Nối (Concatenate) đặc trưng ảnh và metadata
            combined = torch.cat((img_features, meta_features), dim=1)
        else:
            combined = img_features

        return self.classifier(combined)