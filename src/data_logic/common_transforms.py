import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_ONLY_MODES = frozenset({"diag1", "strategy1", "image_only"})
AUGMENTATION_PROFILES = frozenset({"none", "light", "standard"})


def uses_metadata(metadata_mode: str) -> bool:
    return metadata_mode not in IMAGE_ONLY_MODES


def build_dermoscopy_transform(img_size: int, train: bool, augmentation_profile: str = "light") -> A.Compose:
    if not train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unknown augmentation_profile={augmentation_profile!r}. "
            f"Expected one of: {', '.join(sorted(AUGMENTATION_PROFILES))}"
        )

    if augmentation_profile == "none":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    if augmentation_profile == "light":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-15, 15),
                p=0.3,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-45, 45),
            p=0.5,
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.05, 0.1),
            hole_width_range=(0.05, 0.1),
            p=0.3,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
