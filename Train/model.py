import torch.nn as nn
from network import ClfNet

def build_model(args, num_classes=2):
    model = ClfNet(
        featureLength=args.input_feature_length,
        classes=num_classes,
        dropout=args.dropout,
        adapter_type=None,
        adapter_dim=None
    )
    return model.to(args.device)
