from .fusion_head import ConcatenationFusion, DualEmbeddingFusion, MultimodalClassifier

def get_model(config, cat_cardinalities, num_numeric, use_metadata=True):
    """
    Hàm khởi tạo model linh hoạt.
    Tham số use_metadata sẽ ghi đè logic nếu được truyền từ bên ngoài.
    """
    mode = config['METADATA_MODE']
    model_name = config.get('MODEL_NAME', 'tf_efficientnet_b4_ns')
    pretrained = config.get('PRETRAINED', True)
    meta_boost = config.get('METADATA_FEATURE_BOOST', 1.0)

    if mode in ('strategy2', 'concatenation', 'concat'):
        return ConcatenationFusion(
            model_name=model_name,
            pretrained=pretrained,
            cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric,
            num_classes=1,
            use_metadata=use_metadata,
        )

    # Strategy 4: Learnable dual-embedding gating.
    if mode in ('strategy4', 'late_fusion', 'gating'):
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

    # Strategy 1 (image-only) and Strategy 3 (FiLM).
    else:
        actual_use_meta = use_metadata if mode not in ('diag1', 'strategy1', 'image_only') else False
        
        actual_meta_weight = meta_boost if mode in ('full_weighted', 'strategy3_weighted') else 1.0
        
        return MultimodalClassifier(
            model_name=model_name,
            pretrained=pretrained,
            cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric,
            num_classes=1,
            use_metadata=actual_use_meta,
            meta_weight=actual_meta_weight
        )
