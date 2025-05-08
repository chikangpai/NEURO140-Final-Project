import argparse
import enum
import os
from typing import List, Optional

import torch
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download, login

class ModelType(enum.Enum):
    CHIEF = "chief"
    UNI   = "uni"

    def __str__(self):
        return self.value

def parse_model_type(models_str: str) -> List[ModelType]:
    """
    Parse a comma-separated string of model names into a list of ModelType enums.
    """
    names = [n.strip().lower() for n in models_str.split(",")]
    types: List[ModelType] = []
    for n in names:
        try:
            types.append(ModelType(n))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Unknown model: {n}")
    return types

def get_model(model_type: ModelType, hf_token: Optional[str] = None) -> nn.Module:
    """
    Instantiate the requested model.
    Supports only CHIEF and UNI backbones.
    """
    if model_type == ModelType.UNI:
        return _build_uni(hf_token)
    elif model_type == ModelType.CHIEF:
        return _build_chief()
    else:
        # This should never happen if parse_model_type is used correctly
        raise ValueError(f"Unsupported model: {model_type}")

def _build_uni(hf_token: Optional[str]) -> nn.Module:
    """
    Load the UNI model from HuggingFace Hub.
    """
    if hf_token:
        login(token=hf_token)
    cache_dir = os.path.expanduser("~/.cache/uni")
    os.makedirs(cache_dir, exist_ok=True)
    # Download the pretrained checkpoint
    ckpt_path = hf_hub_download(
        repo_id="MahmoodLab/UNI",
        filename="pytorch_model.bin",
        cache_dir=cache_dir,
        force_download=False,
    )
    # Build the ViT backbone without head
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="avg"
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model

def _build_chief() -> nn.Module:
    """
    Build a placeholder CHIEF backbone.
    Here we use a small ViT as a stand-in; replace with your custom architecture if needed.
    """
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=0,
        global_pool="avg"
    )
    return model
