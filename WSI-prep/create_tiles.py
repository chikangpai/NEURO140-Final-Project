"""
Modified from some private repos from our lab, specifically repos from Shih-Yen Lin, Bao Li, and Sophie Tsai.
"""
import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm
import cv2

@dataclass
class PatchPack:
    center: Tuple[int,int]
    images: List[Image.Image]
    tissue_pct: float
    magnifications: List[int]

class H5Saver:
    def __init__(self, path: Path):
        self.f = h5py.File(str(path), 'a')
    def push_pack(self, pack: PatchPack):
        for img,mag in zip(pack.images, pack.magnifications):
            ds = str(mag)
            arr = np.array(img)
            if ds not in self.f:
                self.f.create_dataset(ds, data=arr[np.newaxis],
                                      maxshape=(None,)+arr.shape,
                                      compression='gzip')
            else:
                d=self.f[ds]
                d.resize((d.shape[0]+1,)+d.shape[1:])
                d[-1]=arr
        # metadata
        m=self.f.require_dataset('meta', shape=(0,), maxshape=(None,),
                                  dtype=[('pct','f4')], compression='gzip')
        idx=m.shape[0]
        m.resize((idx+1,))
        m[idx]=pack.tissue_pct
        self.f.flush()
    def keep_top_n(self, n:int):
        m=self.f['meta'][:]
        idxs=np.argsort(m['pct'])[-n:]
        for ds in list(self.f):
            if ds=='meta': continue
            data=self.f[ds][:]
            self.f[ds]=data[idxs]
        self.f['meta']=m[idxs]
        self.f.flush()
    def __len__(self): return len(self.f['meta'])


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--slide_folder', type=Path, required=True)
    p.add_argument('--patch_folder', type=Path, required=True)
    p.add_argument('--patch_size', type=int, default=224)
    p.add_argument('--stride', type=int, default=224)
    p.add_argument('--tissue_threshold', type=float, default=0.8,
                   help='Fractional threshold 0-1')
    p.add_argument('--mags', nargs='+', type=int, default=[40,20])
    p.add_argument('--keep_top_n', type=int, default=None)
    p.add_argument('--n_workers', type=int, default=1)
    return p.parse_args()


def get_slide_id(path: Path) -> str:
    return path.stem


def get_thumbnail(wsi:openslide.OpenSlide, down=16) -> np.ndarray:
    s=wsi.dimensions
    ts=(s[0]//down, s[1]//down)
    return np.array(wsi.get_thumbnail(ts))


def find_tissue_patches(wsi, size, stride, thr) -> List[Tuple[int,int]]:
    thumb=get_thumbnail(wsi)
    hsv=cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv, (0,50,50),(180,255,255))>0
    # simple threshold
    coords=[]
    W,H=wsi.dimensions
    for y in range(0,H-size+1,stride):
        for x in range(0,W-size+1,stride):
            # map to thumb
            tx=int(x/stride); ty=int(y/stride)
            if mask[ty, tx]:
                coords.append((x,y))
    return coords


def extract_patches(slide_path, coord, size, mags) -> PatchPack:
    wsi=openslide.OpenSlide(str(slide_path))
    imgs=[]; pcts=[]; ms=[]
    for mag in mags:
        lvl=0  # assume single level
        img=wsi.read_region(coord, lvl,(size,size)).convert('RGB')
        imgs.append(img)
        gray=np.array(img.convert('L'))
        pcts.append((gray>200).mean())
        ms.append(mag)
    return PatchPack(center=coord, images=imgs,
                     tissue_pct=pcts[0], magnifications=ms)


def process_slide(slide_path, args):
    sid=get_slide_id(slide_path)
    coords=find_tissue_patches(openslide.OpenSlide(str(slide_path)),
                                args.patch_size,args.stride,args.tissue_threshold)
    saver=H5Saver(args.patch_folder/f"{sid}.h5")
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        for pack in tqdm(ex.map(lambda c: extract_patches(slide_path,c,
                                                         args.patch_size,args.mags), coords),
                         total=len(coords)):
            if pack.tissue_pct>=args.tissue_threshold:
                saver.push_pack(pack)
    if args.keep_top_n:
        saver.keep_top_n(args.keep_top_n)
    print(f"Saved {len(saver)} patches for {sid}")


def main():
    args=parse_args()
    args.patch_folder.mkdir(parents=True,exist_ok=True)
    slides=list(Path(args.slide_folder).glob('**/*.svs'))
    for s in slides:
        process_slide(s,args)

if __name__=='__main__':
    main()
