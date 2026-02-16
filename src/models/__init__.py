from .fusion_head import MultimodalClassifier, DualEmbeddingFusion

def get_model(config, cat_cardinalities, num_numeric):
    mode = config['METADATA_MODE']
    if mode == 'late_fusion':
        return DualEmbeddingFusion(
            pretrained=config['PRETRAINED'], cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric, num_classes=1, embed_dim=256
        )
    else:
        meta_weight = config.get('METADATA_FEATURE_BOOST', 1.0) if mode == 'full_weighted' else 1.0
        return MultimodalClassifier(
            pretrained=config['PRETRAINED'], cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric, num_classes=1, use_metadata=(mode != 'diag1'),
            meta_weight=meta_weight
        )