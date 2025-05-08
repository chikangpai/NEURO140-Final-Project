"""
Utilities and dataset classes tailored for TCGA whole-slide image features and patches.
Includes:
 - Success ID loader from success.db or txt
 - Multi-expert TCGA dataset (HDF5-backed)
 - Helper functions for plotting, seeding, metrics
 - Logger interfaces: Wandb and Tensorboard
 - Stain normalizers (Macenko)
 - Transform builders (Albumentations and timm)
"""
import os
import io
import json
import random
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, pil_to_tensor, to_pil_image
from torch.utils.tensorboard import SummaryWriter
import wandb
from omegaconf import DictConfig, OmegaConf

from utils.constants import EMBEDDING_SIZES
from utils.helpers import get_transforms
from timm.data.transforms_factory import create_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== Success IDs Loader ====================

def load_success_ids(feat_folder: str) -> Set[str]:
    """
    Load processed TCGA slide IDs from success.db (preferred) or success.txt.
    Filters slide IDs to TCGA prefix.
    """
    ids = set()
    txt = os.path.join(feat_folder, 'success.txt')
    db = os.path.join(feat_folder, 'success.db')
    if os.path.exists(txt):
        with open(txt) as f:
            ids.update(line.strip() for line in f)
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT slide_id FROM success")
        ids = {row[0] for row in cur.fetchall()}
        conn.close()
    # Only TCGA IDs
    return {sid for sid in ids if sid.startswith('TCGA-')}

# ==================== Dataset ====================

class MultiExpertTCGADataset(Dataset):
    """
    HDF5-backed dataset for TCGA slide embeddings from multiple experts.
    Each sample returns stacked embeddings and padding masks for a slide.
    """
    def __init__(
        self,
        csv_path: str,
        feat_folder: str,
        magnification: int = 40,
        experts: List[str] = ['chief','uni'],
        n_patches: Optional[int] = None,
        random_selection: bool = False,
        get_metadata: bool = False,
    ):
        self.csv = pd.read_csv(csv_path)
        # filter only TCGA slides
        self.csv = self.csv[self.csv['uuid'].str.startswith('TCGA-')]
        self.feat_folder = feat_folder
        self.mag = magnification
        self.experts = [e.upper() for e in experts]
        self.n_patches = n_patches
        self.random = random_selection
        self.meta = get_metadata
        # determine valid slides
        all_ids = set(self.csv['uuid'])
        success = load_success_ids(feat_folder)
        self.slide_ids = sorted(all_ids & success)
        # cache label mapping
        self.labels = {sid: self.csv.loc[self.csv['uuid']==sid,'label'].iloc[0] for sid in self.slide_ids}
        # cache feature counts
        self.counts = {sid: self._count_patches(sid) for sid in self.slide_ids}

    def _count_patches(self, sid: str) -> int:
        path = os.path.join(self.feat_folder, sid, f"{self.mag}x_features.h5")
        with h5py.File(path,'r') as f:
            return f[f"{self.experts[0]}_features"].shape[0]

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int):
        sid = self.slide_ids[idx]
        total = self.counts[sid]
        # select indices
        if self.n_patches is None:
            indices = list(range(total))
        else:
            if self.random:
                indices = random.sample(range(total), min(total, self.n_patches))
            else:
                indices = list(range(min(total, self.n_patches)))
        # load embeddings and masks
        embs, masks = [], []
        hb5 = os.path.join(self.feat_folder, sid, f"{self.mag}x_features.h5")
        with h5py.File(hb5,'r') as f:
            for exp in self.experts:
                arr = torch.from_numpy(f[f"{exp}_features"][()])
                # gather and pad
                sel = []
                mask = []
                for i in indices:
                    if i < arr.shape[0]: sel.append(arr[i]); mask.append(False)
                    else: sel.append(torch.zeros_like(arr[0])); mask.append(True)
                embs.append(torch.stack(sel))
                masks.append(torch.tensor(mask).unsqueeze(-1))
            # optional metadata
            if self.meta and 'metadata' in f:
                md = f['metadata'][()]
                return sid, embs, masks, pd.DataFrame(md), self.labels[sid]
        return sid, embs, masks, self.labels[sid]

# ==================== Helper Functions ====================

def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_to_image(fig, dpi=300) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    fig.clf()
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img).transpose(2,0,1)


def create_roc_curve(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    auc = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC={auc:.2f}')
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend()
    return plot_to_image(fig)


def create_expert_util(names: List[str], utils: np.ndarray, title:str='Expert Utilization') -> torch.Tensor:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.barh(names, utils); ax.set_title(title)
    return to_tensor(Image.fromarray(np.array(fig.canvas.buffer_rgba())))


def compute_expert_variance(weights: torch.Tensor) -> float:
    arr = weights.cpu().numpy() if isinstance(weights,torch.Tensor) else np.array(weights)
    primary = np.argmax(arr,axis=1)
    counts = np.bincount(primary, minlength=arr.shape[1])
    return float(np.var(counts))

# ==================== Loggers ====================

class MetricsLogger:
    def log(self,*args,**kw): raise NotImplementedError
    def log_dict(self,*args,**kw): raise NotImplementedError
    def log_image(self,*args,**kw): raise NotImplementedError
    def set_fold(self,*args,**kw): raise NotImplementedError
    def log_cfg(self,*args,**kw): raise NotImplementedError

class WandbLogger(MetricsLogger):
    def __init__(self, project:str, group:str, tags:List[str], run_name:str):
        self.run = wandb.init(project=project, group=group, name=run_name, tags=tags)
    def log(self, data:Dict[str,Any], step:Optional[int]=None):
        wandb.log(data, step=step)
    def log_image(self, tag:str, img:torch.Tensor):
        wandb.log({tag: wandb.Image(img)})
    def log_cfg(self,cfg:DictConfig): wandb.config.update(OmegaConf.to_container(cfg,resolve=True))
    def set_fold(self, fold:int, cfg:DictConfig):
        self.run.finish(); self.run=wandb.init(name=f'{self.run.name}_fold{fold}')

class TensorboardLogger(MetricsLogger):
    def __init__(self, log_dir:str): self.writer=SummaryWriter(log_dir)
    def log(self, tag:str, val:Any, step:int):
        if isinstance(val,(int,float)): self.writer.add_scalar(tag,val,step)
        else: self.writer.add_histogram(tag,val,step)
    def log_image(self, tag:str, img:torch.Tensor, step:int): self.writer.add_image(tag,img,step)
    def log_cfg(self,cfg:Dict): self.writer.add_text('cfg',json.dumps(cfg,indent=2))
    def set_fold(self, fold:int, _=None): self.writer.flush(); self.writer=SummaryWriter(self.writer.log_dir+f'/fold{fold}')

# ==================== Stain Normalizers ====================

from torchstain.base.normalizers.he_normalizer import HENormalizer

def _cov(x: torch.Tensor) -> torch.Tensor:
    m = torch.mean(x, dim=0, keepdim=True)
    return (x - m).T @ (x - m) / (x.shape[0]-1)

class TorchMacenkoNormalizer(HENormalizer):
    def __init__(self): super().__init__()

class TorchBatchMacenkoNormalizer(torch.nn.Module):
    def __init__(self, Io=240, alpha=1, beta=0.15, source_thumbnail:Optional[str]=None, source_thumbnail_mask:Optional[str]=None):
        super().__init__(); self.Io, self.alpha, self.beta = Io, alpha, beta
        self.norm = TorchMacenkoNormalizer()
        # fit reference if thumbnail provided
        if source_thumbnail:
            thumb = Image.open(source_thumbnail).convert('RGB')
            t = torch.tensor(np.array(thumb)).permute(2,0,1)
            self.HE, _ = self.norm.fit_source(t, Io, alpha, beta)
        else: self.HE = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape if x.dim()==4 else (1,)+x.shape
        x_ = x if x.dim()==3 else x
        y,_,_ = self.norm.normalize(x_, Io=self.Io, alpha=self.alpha, beta=self.beta, HE=self.HE, stains=False)
        return y if x.dim()==3 else y.reshape(B,C,H,W)

# ==================== Transforms ====================

def get_transforms_albumentation(train:bool=False):
    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    def tf(imgs):
        imgs = imgs if isinstance(imgs,list) else [imgs]
        n=len(imgs)
        keys={f'image{i}':'image' for i in range(1,n)}
        comp = A.Compose([A.Resize(224,224), A.RandomResizedCrop(224,224) if train else None, A.HorizontalFlip() if train else None, A.Normalize(mean,std), ToTensorV2()], additional_targets=keys)
        arrs=[np.array(im) for im in imgs]
        data={'image':arrs[0], **{f'image{i}':arrs[i] for i in range(1,n)}}
        out=comp(**{k:v for k,v in data.items() if v is not None})
        return [out[k] for k in out if 'image' in k][0] if n==1 else [out[k] for k in sorted(out) if 'image' in k]
    return tf

def get_transforms_timm(cfg:Dict[str,Any]={}):
    return create_transform(**cfg)  
