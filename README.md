# NEURO140-Final-Project

An end-to-end pipeline for preparing whole-slide images (WSIs) from TCGA, extracting patch embeddings with pretrained vision models, and loading them for downstream machine-learning experiments.  

---

## 📁 Folder Structure

```

.
├── WSI-prep/             # TCGA WSI tiling & embedding pipeline
│   ├── slides/           # raw SVS files (downloaded from GDC)
│   ├── tiles/            # HDF5 patch-banks (output of tiling)
│   ├── embeddings/       # precomputed patch embeddings (.npy or .pt)
│   ├── utils/            # helper scripts (I/O, stain-norm, etc.)
│   └── run\_pipeline.py   # orchestrates tiling → embedding → saving
├── models/               # (future) model definitions & training code
├── experiments/          # (future) notebooks, configs, W\&B logs
└── README.md             # this file

````

---

## 🔍 Attributions & Origins

- **Tiling & feature-extraction logic**  
  Adapted from private lab repos by **Shih-Yen Lin**, **Bao Li**, and **Sophie Tsai**.  
- **Patch embedding code**  
  Based on the open-source [Owkin HistoSSLscaling](https://github.com/owkin/HistoSSLscaling) project.  
- **My own contributions in WSI preparations**  
  - TCGA download scripts & metadata parsing  
  - Unified pipeline (tiling → embedding → saving)  
  - Integration of stain-normalization, slide filtering, and tile subsampling  
  - Logging experiments to Weights & Biases (W&B)  

---

## ⚙️ Prerequisites

Make sure you have **Python 3.10+**:

```bash
conda create -n tcga-py310 python=3.10
conda activate tcga-py310
conda install -c conda-forge openslide-python openslide
pip install \
    torch torchvision timm albumentations \
    opencv-python-headless h5py pillow pandas \
    numpy scikit-learn tqdm wandb omegaconf torchstain
````

---

## 🚀 Step-by-Step Usage

### 1. Download WSIs

1. Obtain TCGA slide IDs from the GDC portal.
2. Place it in the desired path.

### 2. Tile WSIs → HDF5 patch banks

```bash
cd WSI-prep
python run_pipeline.py \
  --slide-dir slides/ \
  --tile-output-dir tiles/ \
  --tile-size 224 \
  --stride 224 \
  --stain-norm macenko
```

* **Output**: one HDF5 file per slide, containing RGB patches.

### 3. Embed patches with pretrained models

```bash
python create_features.py
  --model chief \
  --batch-size 256 \
  --output-dir output_path\
  ...
```


---

## 🏗 Model Framework (scaffold)

The **`models/`** folder will house your classification, survival, and fairness-aware code:

* **Network definitions**

  * `ClfNet` (classification)
* **Training entrypoint**

  * `main_genetic.py` accepts arguments via `framework.parse_args()`
  * Supports flags: `--skip_existing`, `--inference_only`, `--max_train_tiles`, etc.

### Example training command

```bash
python main_genetic.py \
  --cancer brca \
  --task 4 \
  --partition 2 \
  --train_method baseline \
  --model_path /path/to/models/ \
  --fair_attr '{"age":["old","young"]}' \
  --skip_existing \
  --epochs 100 \
  --batch_size 16 \
  --eval_batch_size 4 \
  --device cuda
```

---

---

## 📄 License & Acknowledgments

This project builds on private lab code (Lin, Li, Tsai) and the Owkin HistoSSLscaling repo.
Please cite those sources if you use this work in publication.

```plaintext
@software{owkin_histolssl,
  author = {Owkin},
  title = {HistoSSLscaling},
  year = {2023},
  url = {https://github.com/owkin/HistoSSLscaling}
}
```

---