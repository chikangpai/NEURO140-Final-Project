"""
Modified from some private repos from our lab, specifically repos from Shih-Yen Lin, Bao Li, and Sophie Tsai.
"""
import argparse
import glob
import math
import os
import sqlite3
import stat
import time
import traceback
from functools import wraps, partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import openslide
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm import tqdm

from models.library import get_model, parse_model_type
from utils.transforms import TorchBatchMacenkoNormalizer, get_transforms_timm as get_transforms

# Defaults
DEFAULT_MODELS = ["chief", "uni"]
H5_COMPRESSION = "gzip"
H5_COMPRESSION_OPTS = 9
TQDM_MIN_INTERVAL = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from patches and save in HDF5/PT formats"
    )
    parser.add_argument("--patch_folder", type=Path, required=True,
                        help="Root folder containing coords/*.h5 files")
    parser.add_argument("--wsi_folder", type=Path, required=True,
                        help="Root folder containing WSI files (.svs, .ndpi, .tiff)")
    parser.add_argument("--feat_folder", type=Path, required=True,
                        help="Output folder for features: <feat_folder>/<model>/<mag>X")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for DataLoader")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Number of DataLoader workers")
    parser.add_argument("--n_parts", type=int, default=1,
                        help="Split workload into parts for parallel jobs")
    parser.add_argument("--part", type=int, default=0,
                        help="Index of part to process (0-based)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device (cuda or cpu)")
    parser.add_argument("--target_mag", type=int, default=40,
                        help="Desired magnification for patch extraction")
    parser.add_argument("--stain_norm", action="store_true",
                        help="Enable Macenko stain normalization")
    parser.add_argument("--img_format", choices=["wsi","tiff"], default="wsi",
                        help="Input image format; 'tiff' uses fixed magnification")
    parser.add_argument("--tiff_mag", type=float, default=40.0,
                        help="Magnification for TIFF images")
    parser.add_argument("--wsi_mag", type=int, default=-1,
                        help="Override WSI objective power if >0")
    parser.add_argument("--models", type=parse_model_type,
                        default=DEFAULT_MODELS,
                        help="Models to extract features from (default: chief, uni)")
    return parser.parse_args()


class WSIDataset(Dataset):
    """
    Reads coordinates from HDF5 and extracts patches at desired magnification.
    """
    def __init__(
        self,
        args: Any,
        h5_path: Path,
        wsi: openslide.OpenSlide,
        transforms: nn.Module,
    ):
        self.args = args
        self.h5_path = h5_path
        self.wsi = wsi
        self.transforms = transforms
        self.stain_norm = args.stain_norm

        # load coords and metadata
        with h5py.File(self.h5_path, 'r') as f:
            self.coords = f['coords'][:]  # Nx2 array
            md = f['metadata']
            self.patch_size = int(md['patch_size'][0])
        self.length = len(self.coords)

        # determine read level and scaling
        self._init_read_params()
        if self.stain_norm:
            self.normalizer = TorchBatchMacenkoNormalizer(
                source_thumbnail=None, source_thumbnail_mask=None
            )

    def _init_read_params(self):
        # determine highest mag
        if self.args.img_format == 'wsi':
            power = float(self.wsi.properties.get('openslide.objective-power', self.args.wsi_mag))
            if self.args.wsi_mag > 0:
                power = self.args.wsi_mag
            # map native magnifications to levels
            native = {round(power/ds,2): lvl for lvl, ds in enumerate(self.wsi.level_downsamples)}
            if self.args.target_mag in native:
                self.level = native[self.args.target_mag]
                self.scale = 1.0
            else:
                # choose nearest higher
                higher = [m for m in native if m>self.args.target_mag]
                src = max(higher) if higher else max(native)
                self.level = native[src]
                self.scale = src / self.args.target_mag
        else:
            self.level = 0
            self.scale = self.args.tiff_mag / self.args.target_mag

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        x, y = tuple(self.coords[idx])
        size = self.patch_size
        if self.scale != 1.0:
            sz = int(size * self.scale)
            img = self.wsi.read_region((x,y), self.level, (sz,sz))
            img = img.resize((size,size), Image.BILINEAR)
        else:
            img = self.wsi.read_region((x,y), self.level, (size,size))
        img = img.convert('RGB')
        if self.stain_norm:
            t = pil_to_tensor(img)
            t = self.normalizer(t).type_as(t)
            img = to_pil_image(t)
        tensor = self.transforms(img)
        return tensor, np.array([x,y])


# retry decorator

def retry(max_retries=3, delay=5, exceptions=(Exception,)):
    def deco(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if i==max_retries-1:
                        raise
                    time.sleep(delay)
        return wrapper
    return deco


@retry(max_retries=5, delay=2, exceptions=(OSError,))
def load_models(args: Any) -> Dict[str, nn.Module]:
    models = {}
    for name in args.models:
        models[name] = get_model(args, name).to(args.device)
    return models


def initialize_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        mode = os.stat(db_path).st_mode
        os.chmod(db_path, mode | stat.S_IRGRP | stat.S_IWGRP)
    except PermissionError:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS success (
            slide_id TEXT PRIMARY KEY,
            models TEXT
        )""")
    conn.commit()
    conn.close()


def load_success(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query("SELECT * FROM success", conn)
    conn.close()
    return df


def update_success(db_path: Path, slide_id: str, models: List[str]) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT models FROM success WHERE slide_id=?", (slide_id,))
    row = cur.fetchone()
    if row:
        existing = set(row[0].split(','))
        updated = sorted(existing.union(models))
        cur.execute("UPDATE success SET models=? WHERE slide_id=?",
                    (','.join(updated), slide_id))
    else:
        cur.execute("INSERT INTO success (slide_id, models) VALUES (?,?)",
                    (slide_id, ','.join(models)))
    conn.commit()
    conn.close()


def save_hdf5(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), 'a') as f:
        for k,v in data.items():
            arr = np.asarray(v)
            if k not in f:
                d=f.create_dataset(k, data=arr,
                                  maxshape=(None,)+arr.shape[1:],
                                  chunks=True,
                                  compression=H5_COMPRESSION,
                                  compression_opts=H5_COMPRESSION_OPTS)
            else:
                d=f[k]
                old=d.shape[0]; new=old+arr.shape[0]
                d.resize((new,)+d.shape[1:])
                d[old:]=arr


def get_features(batch: torch.Tensor, models: Dict[str, nn.Module], device: str) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        return {name: m(batch.to(device)).cpu() for name, m in models.items()}


def main():
    args = parse_args()
    args.feat_folder.mkdir(parents=True, exist_ok=True)
    db = args.feat_folder / ("success.db" if not args.stain_norm else "success_stainnorm.db")
    initialize_db(db)

    # list coord files
    all_h5 = sorted((args.patch_folder / 'coords').glob('*.h5'))
    # split parts
    total = len(all_h5)
    per_part = math.ceil(total/args.n_parts)
    selected = all_h5[args.part*per_part : min((args.part+1)*per_part, total)]
    slide_ids = [p.stem for p in selected]

    models = load_models(args)
    # Create model output directories
    for m in args.models:
        (args.feat_folder/m/f"{args.target_mag}X").mkdir(parents=True, exist_ok=True)
    succ = load_success(db)
    need = set(args.models)

    for idx, sid in enumerate(slide_ids):
        if sid in succ['slide_id'].values:
            done = set(succ.loc[succ['slide_id']==sid,'models'].iloc[0].split(','))
            if need.issubset(done):
                continue
        try:
            # load wsi
            paths = glob.glob(str(args.wsi_folder / f"**/{sid}.*"), recursive=True)
            wsi = openslide.OpenSlide(paths[0])
            # dataset and loader
            ds = WSIDataset(args, args.patch_folder/'coords'/f"{sid}.h5", wsi, get_transforms())
            dl = DataLoader(ds, batch_size=args.batch_size,
                            num_workers=args.n_workers,
                            collate_fn=lambda b: (torch.cat([x for x,_ in b]),
                                                 np.vstack([y for _,y in b])))
            # extract and save
            for batch, coords in tqdm(dl, mininterval=TQDM_MIN_INTERVAL):
                out = get_features(batch, models, args.device)
                for name,ft in out.items():
                    save_hdf5(args.feat_folder/name/f"{args.target_mag}X"/"features.h5",
                              {'features':ft.numpy(), 'coords':coords})
            update_success(db, sid, args.models)
        except Exception as e:
            print(f"Error {sid}: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    main()


