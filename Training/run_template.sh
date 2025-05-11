#!/bin/bash
#SBATCH -c 4                               # Request one core
#SBATCH -t 24:00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_yu
#SBATCH --account= your account for gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -o ./logs/finetune_TCGA_mutation_auroc/CHIEF/finetune_TCGA_mutation_auroc%A_%a.log                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./logs/finetune_TCGA_mutation_auroc/CHIEF/finetune_TCGA_mutation_auroc%A_%a.log                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --array=0-30
#               /\
#              /||\
#               ||     
#              You can change the cancer types by changing the number after the dash in the --array flag
#              See CANCER below for the list of cancer types

# IDX=$((SLURM_ARRAY_TASK_ID))

IDX=$((SLURM_ARRAY_TASK_ID))
## cancer types (10 per line)
declare -a CANCER=("brca" "coadread" "gbm"  "kich" "kirc"   "kirp" "lgg"  "luad" "lusc" "ov" \
                   "ucec" "prad"     "thca" "hnsc" "stad"   "skcm" "blca" "sarc" "lihc" "cesc"  \
                   "paad" "tgct"     "esca" "pcpg" "acc"    "thym" "meso" "ucs"   "uvm" "chol" \
                   "dlbc")
## dictionay for feature lengths
declare -A INPUT_FEATURE_LENGTHS=(\
    ["CHIEF"]=768 \
    ["UNI"]=1024 \
    ["GIGAPATH"]=1536 \
    ["VIRCHOW2"]=2560)

partition=2

# 
############################################################################################################
#### Dataset and model configurations
cancer=${CANCER[$IDX]} # cancer type
# SENSITIVE='{"Race Category": ["White", "Black or African American"]}'  # either {"Race Category": ["White", "Black or African American"]} OR {"Sex": ["Female", "Male"]}
# SENSITIVE='{"Sex": ["Female", "Male"]}'  # either {"Race Category": ["White", "Black or African American"]} OR {"Sex": ["Female", "Male"]}
SLIDE_TYPE="PM" # either PM or FS
DATA_SOURCE="TCGA"
FOUNDATION_MODEL="CHIEF" # CHIEF, GIGAPATH, UNI, VIRCHOW2
MAGNIFICATION=20
INPUT_FEATURE_LENGTH=${INPUT_FEATURE_LENGTHS[$FOUNDATION_MODEL]}
INFERENCE_ONLY=false  # turn to true if doing inference only
CUTOFF_METHOD=none # method for selecting cutoff for binary classification
TASK=4  # 1:cancer classification, 2:tumor detection, 3:survival prediction, 4:genetic classification

############################################################################################################
#### Training configurations
EPOCHS=100 # number of epochs for training
N_WORKERS=4
## whether to use early stopping
USE_EARLY_STOPPING=true
PATIENCE=10
## whether to randomly subsample the tiles during training
LIMIT_TILES=true  ## if true, limit the number of tiles to per slide during training to MAX_TILES 
MAX_TILES=2000
# Change the model path to your own storage path
MODEL_PATH="./mutation_models/${SEN}/${DATA_SOURCE}/${FOUNDATION_MODEL}/${SLIDE_TYPE}/" # storing interence results and model weight s
TRAIN_METHOD=auroc-adapter-bottleneck-64
# Or auroc-adapter-bottleneck-64-no-pretrained or baseline.
# for cancer in "${CANCER[@]}";

# do for curr in ${CURR[@]};
python /home/chp6257/PEFT/QALY-Tuning/main_genetic.py --cancer $cancer \
                  --model_path="$MODEL_PATH" \
                  --partition=$partition \
                  --train_method=$TRAIN_METHOD \
                  --task=$TASK \
                  --lr=1e-4 \
                  --dropout=0.25 \
                  --seed=0 \
                  --epochs=$EPOCHS \
                  --n_workers=$N_WORKERS \
                  --batch_size=16 \
                  --eval_batch_size=4 \
                  --acc_grad=2 \
                  --scheduler_step=1 \
                  --scheduler_gamma=0.955 \
                  --device="cuda" \
                  --data_source="$DATA_SOURCE" \
                  --cutoff_method=$CUTOFF_METHOD \
                  --input_feature_length="$INPUT_FEATURE_LENGTH" \
                  --foundation_model="$FOUNDATION_MODEL" \
                  --slide_type=$SLIDE_TYPE \
                  --magnification=$MAGNIFICATION \
                  --selection="AUROC" \
                  --stain_norm \
                  $( [ "$INFERENCE_ONLY" = true ] && echo "--inference_only" ) \
                  $( [ "$LIMIT_TILES" = true ] && echo "--max_train_tiles=$MAX_TILES" ) \
                  $( [ "$USE_EARLY_STOPPING" = true ] && echo "--patience=$PATIENCE" ) 

# 
