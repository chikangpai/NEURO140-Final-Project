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
│   ├── slides/                # raw files stored here
│   ├── tiles/                 # HDF5 patch banks (output of tiling)
│   ├── coords/                # optional: stored tile coordinates(Large, store it in lab storage)
│   ├── features/              # HDF5 embeddings per slide(Large, store it in lab storage)
│   ├── utils/                 # I/O, stain-norm, thumbnail helpers
│   ├── extract_template.sh    # I use this sh file to submit a job on my o2 server to run extraction 
│   ├── create\_tiles.py        # build tiles → HDF5
│   └── create\_features.py     # extract by CHIEF/UNI → HDF5/.npy
├── Training/         # TCGA gene-mutation training. After storing slides, tiles and features in the right place
│   ├── config/                   # training and dataset configs
│   ├── main_genetic.py           # main training entrypoint
│   ├── framework.py              # arguments and training functions used in training
│   ├── dataset.py                # Deal with tcga datasets.
│   ├── util.py                   # Helpers to save some arguments.
│   ├── run_template.sh           # I use this sh file to submit a job on my o2 server to run the training 
│   └── network.py                # ClfNet and MIL attentions, adpaters.
└── README.md                     # this file

````

---

## 🔍 Attributions

- **Tiling & feature-extraction**  
  Integrated some private lab repos (Contributors: Shih-Yen Lin, Bao Li, Sophie Tsai). I develop it specifically in TCGA datasets and in gene mutation tasks. I also implemented the adapter/classifier logic.
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
Run the sh file template, make sure you got enough storage for the slides and patches, and modify the paths accordingly.

---

### 2. Embed patches with CHIEF or UNI(baseline)

Run the sh file template also, make sure you got enough storage for the slides and embeddings, and modify the paths accordingly.

---

### 3. Train gene-mutation classifier

In the multi-instance learning part, you can choose whether to use adapter. Simply change the "TRAIN_METHOD" in the sh file.

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
