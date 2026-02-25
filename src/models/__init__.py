from .fusion_head import MultimodalClassifier, DualEmbeddingFusion

def get_model(config, cat_cardinalities, num_numeric, use_metadata=True):
    """
    Hàm khởi tạo model linh hoạt.
    Tham số use_metadata sẽ ghi đè logic nếu được truyền từ bên ngoài.
    """
    mode = config['METADATA_MODE']
    model_name = config.get('MODEL_NAME', 'tf_efficientnet_b4_ns')
    pretrained = config.get('PRETRAINED', True)
    meta_boost = config.get('METADATA_FEATURE_BOOST', 1.0)

    # 1. Nếu dùng kiến trúc Late Fusion (Dual Embedding)
    if mode == 'late_fusion':
        return DualEmbeddingFusion(
            model_name=model_name,
            pretrained=pretrained,
            cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric,
            num_classes=1,
            embed_dim=256,
            meta_weight=meta_boost,
            use_metadata=use_metadata  # Truyền flag ngắt metadata
        )

    # 2. Các mode còn lại (diag1, full, full_weighted) dùng MultimodalClassifier
    else:
        # Nếu mode là diag1, ép use_metadata = False bất kể tham số truyền vào
        actual_use_meta = use_metadata if mode != 'diag1' else False
        
        # Chỉ dùng meta_weight boost khi ở chế độ weighted
        actual_meta_weight = meta_boost if mode == 'full_weighted' else 1.0
        
        return MultimodalClassifier(
            model_name=model_name,
            pretrained=pretrained,
            cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric,
            num_classes=1,
            use_metadata=actual_use_meta,
            meta_weight=actual_meta_weight
        )