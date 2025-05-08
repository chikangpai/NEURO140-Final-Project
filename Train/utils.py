import os, json
import numpy as np
from sklearn.metrics import roc_auc_score

def save_args(args, outdir, fname="args.json"):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, fname), "w") as f:
        json.dump(vars(args), f, indent=2)

def get_predictions(probs, threshold=None, method="none"):
    if threshold is None or method=="none":
        return (probs > 0.5).astype(int)
    # you can add other cutoff methods here
    return (probs > threshold).astype(int)

# expose roc
roc_auc_score = roc_auc_score
