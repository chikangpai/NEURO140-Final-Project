#!/bin/bash
#SBATCH -c 4                   # CPU cores
#SBATCH -t 03:00:00            # hh:mm:ss
#SBATCH -p gpu                 # partition / queue
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o logs/feat_ext_%A_%a.out
#SBATCH -e logs/feat_ext_%A_%a.err
#SBATCH --array=0-2            # one job per cohort

# ---------------- user settings ----------------
COHORTS=( "TCGA-BRCA-PM" "TCGA-LGG-PM" "TCGA-THCA-PM" )
MODEL_LIST="chief"            # or chief,uni
OUTPUT_ROOT="$HOME/wsi_features_adapter"     # destination for HDF5s

# (optional) HF token for gated models (UNI, Virchow…) – leave blank if not needed
HF_TOKEN=""

# ---------------- derived vars -----------------
COHORT=${COHORTS[$SLURM_ARRAY_TASK_ID]}
SLIDE_DIR="/path/to/${COHORT}/svs"                  # <-- edit
TILE_DIR="/path/to/${COHORT}/tiles"                 # <-- edit
FEAT_DIR="${OUTPUT_ROOT}/${COHORT}"
mkdir -p "${FEAT_DIR}"

# ---------------- environment -------------------
module purge
# module load <cuda> <gcc> …    # as needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tcga-py310

# model checkpoints (only needed for CHIEF / CTransPath)
export CHIEF_PTH="/path/to/chief.pth"       # <-- edit
export CTRANS_PTH="/path/to/ctranspath.pth" # <-- edit

# ---------------- run extractor -----------------
python WSI-prep/create_features.py \
  --patch_folder   "${TILE_DIR}" \
  --wsi_folder     "${SLIDE_DIR}" \
  --feat_folder    "${FEAT_DIR}" \
  --models         "${MODEL_LIST}" \
  --device         cuda \
  --target_mag     20 \
  --stain_norm \
  --adapter_type   bottleneck \           # NEW ⚡
  --adapter_dim    64 \                   # NEW ⚡
  --hf_token       "${HF_TOKEN}"
