
from .fusion_head import MultimodalModel, DualEmbeddingFusion

def get_model(config, cat_cardinalities, num_numeric):
    backbone_type = config.get('BACKBONE_TYPE', 'efficientnet')
    mode = config.get('METADATA_MODE', 'diag1')
from .fusion_head import MultimodalClassifier, DualEmbeddingFusion  # Đã đổi tên class


def get_model(config, cat_cardinalities, num_numeric):
    mode = config['METADATA_MODE']
    model_name = config.get('MODEL_NAME', 'efficientnet_b4')  # Lấy tên từ config

    if mode == 'late_fusion':
        # Nếu bạn dùng DualEmbeddingFusion, hãy đảm bảo class này trong fusion_head.py 
        # cũng đã được sửa để nhận backbone_type tương tự MultimodalModel
        return DualEmbeddingFusion(
            backbone_type=config['BACKBONE_TYPE'], 
            pretrained=config.get('PRETRAINED', True), 
            cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric, 
            num_classes=1, 
            embed_dim=256
        )
    else:
        meta_weight = config.get('METADATA_FEATURE_BOOST', 1.0) if mode == 'full_weighted' else 1.0
        return MultimodalModel(
            backbone_type=config['BACKBONE_TYPE'],  # Truyền vào đây
            pretrained=config.get('PRETRAINED', True), 
            cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric, 
            num_classes=1, 
        )