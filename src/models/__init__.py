from .fusion_head import MultimodalClassifier, DualEmbeddingFusion  # Đã đổi tên class


def get_model(config, cat_cardinalities, num_numeric):
    mode = config['METADATA_MODE']
    model_name = config.get('MODEL_NAME', 'efficientnet_b4')  # Lấy tên từ config

    if mode == 'late_fusion':
        return DualEmbeddingFusion(
        pretrained=config['PRETRAINED'],
        cat_cardinalities=cat_cardinalities,
        num_numeric=num_numeric,
        num_classes=1,
        embed_dim=256,
        meta_weight=config.get('METADATA_FEATURE_BOOST', 1.0)
    )

    else:
        meta_weight = config.get('METADATA_FEATURE_BOOST', 1.0) if mode == 'full_weighted' else 1.0
        # Gọi class mới MultimodalClassifier
        return MultimodalClassifier(
            model_name=model_name,  # Truyền ResNet hay EfficientNet vào đây
            pretrained=config['PRETRAINED'],
            cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric,
            num_classes=1,
            use_metadata=(mode != 'diag1'),
            meta_weight=meta_weight
        )