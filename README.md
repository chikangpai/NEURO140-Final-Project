# NEURO140-Final-Project

An end-to-end pipeline for  
1. tiling TCGA whole-slide images (WSIs),  
2. extracting patch embeddings with pretrained vision backbones, and  
3. training a binary gene-mutation classifier.(with or without adpaters)

---

## 📁 Repository Layout

```

.
├── WSI-prep/                  # TCGA WSI tiling & embedding
│   ├── slides/                # raw SVS files (download from GDC)
│   ├── tiles/                 # HDF5 patch banks (output of tiling)
│   ├── coords/                # optional: stored tile coordinates(Large, store it in lab storage)
│   ├── features/              # HDF5 or .npy embeddings per slide(Large, store it in lab storage)
│   ├── utils/                 # I/O, stain-norm, thumbnail helpers
│   ├── create\_tiles.py        # build tiles → HDF5
│   └── create\_features.py     # extract by CHIEF/UNI → HDF5/.npy
├── train/                     # TCGA gene-mutation training
│   ├── main.py                # main training entrypoint
│   ├── args.py                # arguments used in training
│   ├── data.py                # Deal with tcga datasets.
│   ├── utils.py                # Helpers to save some arguments.
│   ├── train.py                # Training functions.
│   └── model.py               # The simple classifier models.
└── README.md                  # this file

````

---

## 🔍 Attributions

- **Tiling & feature-extraction**  
  Adapted and integrated some private lab repos (Contributors: Shih-Yen Lin, Bao Li, Sophie Tsai). I refractored and integrated it, and use it in TCGA datasets, refractored and add adapter logic.
- **Patch embeddings**  
  Leveraging open-source [Owkin HistoSSLscaling](https://github.com/owkin/HistoSSLscaling).  
---

## ⚙️ Prerequisites

```bash
conda install -c conda-forge openslide-python openslide
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Tile WSIs → HDF5 patch banks

```bash
cd WSI-prep
python create_tiles.py \
  --slide_folder slides/ \
  --patch_folder tiles/ \
  --patch_size 224 \
  --stride 224 \
  --tissue_threshold 0.8 \
  --mags 40 20
```

**Output**:
`tiles/TCGA-XX-YYYY.h5` containing

* datasets `40`, `20` (RGB patches)
* `meta` (tissue percentage)

---

### 2. Embed patches with CHIEF or UNI(baseline)

```bash
cd WSI-prep
python create_features.py \
  --patch_folder tiles/ \
  --wsi_folder slides/ \
  --feat_folder embeddings/ \
  --models chief,uni \
  --batch_size 256 \
  --device cuda
```

**Output** per slide:

```
embeddings/TCGA-XX-YYYY/40x_CHIEF.npy
embeddings/TCGA-XX-YYYY/40x_UNI.npy
success.db
```

### 2.1 Extract features with adapters

The codebase supports bottleneck adapters for efficient fine-tuning. To use adapters:

1. Ensure the adapter code is present:
   - `models/adapter.py` (bottleneck block)
   - `inject_adapter()` in `models/library.py`

2. Run feature extraction with adapters:

```bash
cd WSI-prep
python create_features.py \
  --patch_folder tiles/ \
  --wsi_folder slides/ \
  --feat_folder embeddings/ \
  --models chief \
  --device cuda \
  --target_mag 20 \
  --stain_norm \
  --adapter_type bottleneck \
  --adapter_dim 64
```

**What happens?**
- CHIEF weights stay frozen
- Each Transformer block gets a 64-d bottleneck adapter
- Only adapter parameters are saved in HDF5s (under dataset name `adapter_weights`)

---

### 3. Train gene-mutation classifier

```bash
cd train
python main.py \
  --cancer BRCA \
  --gene  TP53 \
  --feat_folder ../WSI-prep/embeddings \
  --model CHIEF \
  --mag     40 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --epochs  20 \
  --lr      1e-4 \
  --device  cuda
```

---

## 📄 License & Citation

This work builds on private-lab code (Lin, Li, Tsai) and Owkin's HistoSSLscaling. Please cite:

```bibtex
@software{owkin_histolssl,
  author = {Owkin},
  title  = {HistoSSLscaling},
  year   = {2023},
  url    = {https://github.com/owkin/HistoSSLscaling}
}
```
