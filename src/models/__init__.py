# from .fusion_head import EfficientNetB4_Multimodal, DualEmbeddingFusion
#
# def get_model(config, cat_cardinalities, num_numeric):
#     mode = config['METADATA_MODE']
#     if mode == 'late_fusion':
#         return DualEmbeddingFusion(
#             pretrained=config['PRETRAINED'], cat_cardinalities=cat_cardinalities,
#             num_numeric=num_numeric, num_classes=1, embed_dim=256
#         )
#     else:
#         meta_weight = config.get('METADATA_FEATURE_BOOST', 1.0) if mode == 'full_weighted' else 1.0
#         return EfficientNetB4_Multimodal(
#             pretrained=config['PRETRAINED'], cat_cardinalities=cat_cardinalities,
#             num_numeric=num_numeric, num_classes=1, use_metadata=(mode != 'diag1'),
#             meta_weight=meta_weight
#         )
# File: src/models/__init__.py
from .fusion_head import EfficientNetB4_Multimodal, DualEmbeddingFusion, \
    ResNet50_Multimodal  # <--- Import thêm ResNet50


def get_model(config, cat_cardinalities, num_numeric):
    mode = config['METADATA_MODE']
    # Lấy tên model từ config, mặc định vẫn là efficientnet để không lỗi code cũ
    model_name = config.get('MODEL_NAME', 'efficientnet')

    # 1. Nếu dùng Late Fusion (Chưa hỗ trợ ResNet ở code này, giữ nguyên)
    if mode == 'late_fusion':
        return DualEmbeddingFusion(
            pretrained=config['PRETRAINED'], cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric, num_classes=1, embed_dim=256
        )

    # 2. Chế độ Multimodal thường
    else:
        meta_weight = config.get('METADATA_FEATURE_BOOST', 1.0) if mode == 'full_weighted' else 1.0

        # --- LOGIC CHỌN MODEL ---
        if model_name == 'resnet50':
            print("⚡ Đang khởi tạo Model: ResNet50_Multimodal")
            return ResNet50_Multimodal(
                pretrained=config['PRETRAINED'], cat_cardinalities=cat_cardinalities,
                num_numeric=num_numeric, num_classes=1, use_metadata=(mode != 'diag1'),
                meta_weight=meta_weight
            )
        else:
            print("⚡ Đang khởi tạo Model: EfficientNetB4_Multimodal")
            return EfficientNetB4_Multimodal(
                pretrained=config['PRETRAINED'], cat_cardinalities=cat_cardinalities,
                num_numeric=num_numeric, num_classes=1, use_metadata=(mode != 'diag1'),
                meta_weight=meta_weight
            )