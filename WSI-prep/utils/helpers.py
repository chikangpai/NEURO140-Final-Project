from timm.data.transforms_factory import create_transform

def get_transforms(size: int = 224, train: bool = False):
    """
    Simple TIMM transform wrapper (ImageNet stats, bicubic).
    Returns a callable that converts PIL → torch.Tensor.
    """
    cfg = {
        "input_size": (3, size, size),
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std":  (0.229, 0.224, 0.225),
        "crop_pct": 1.0,
        "crop_mode": "center",
        "prob": 0.0 if not train else 0.5,
    }
    return create_transform(**cfg)
