import argparse
import os
import json
import wandb
import torch
import glob
from pathlib import Path
from tqdm import tqdm
from network import ClfNet, WeibullModel, MLP
from util import *
from os.path import join
import typing
from typing import List, Union
from dataset import *
from sklearn.metrics import roc_auc_score

from bootstrap_significant_test.bootstrap_TCGA_bias_test import CV_bootstrap_bias_test
from bootstrap_significant_test.bootstrap_TCGA_improvement_test import CV_bootstrap_improvement_test
import yaml

N_FOLDS = 4

FOUNDATION_MDL_FEATURE_TYPE_MAP = {
    'CHIEF': 'tile',
    'UNI': 'tile',
    'GIGAPATH': 'tile',
    'VIRCHOW2': 'tile',
    'TITAN': 'slide'
}
    

def parse_args(input_args=None, print_args=False):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancer",
        nargs='+',
        default=None,
        required=True,
        help="Cancers are the targets for this task.",
    )
    parser.add_argument(
        "--clinical_information_path",
        type=str,
        default='clinical_information',
        help="clinical information path",
    )
    parser.add_argument(
        "--foundation_model",
        default='CHIEF',
        help="foundation model used: CHIEF, UNI, GIGAPATH, VIRCHOW2, TITAN",
    )
    parser.add_argument(
        "--max_train_tiles",
        default=None,
        type=int,
        help="Maximum number of tiles to use for training. If None, all samples are used.",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default='tile',
        choices=['tile', 'slide'],
        help="Type of feature used for training. Either tile or slide.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers for data loading.")

    parser.add_argument(
        "--task",
        default=1,
        help="""
        Type of downstream task:
            - basic tasks are in number format:
                - 1:cancer classification, 2:tumor detection, 3:survival prediction, 4:genetic classification
            - Also support custom tasks in string format. in this case, the task name specifies the label column in the dataset.
                (please make sure the label column is in the clinical information file)
                - e.g. "ER" for estrogen receptor status prediction
                - e.g. "BRCA1" for BRCA1 mutation prediction
        """,
    )
    parser.add_argument(
        "--genes",
        nargs='+',
        default=None,
        help="For mutation prediction task, specify the genes to classify. If not specified, all genes will be used.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/",
        help="Path to load model weights.",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        default=None,
        help="Path to save the inference results. If not specified, the model is not saved to --model_path.",
    )
        
    parser.add_argument(
        "--weight_path",
        type=str,
        default="",
        help="Path to stage 1 pretrained weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs for training."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Patience for early stopping. If not specified, no early stopping is used."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for sampling images."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation. If not specified, the same as batch_size.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--curr_fold",
        type=int,
        default=None,
        help="For k-fold experiments, current fold."
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=1,
        help="Data partition method:'1:train/valid/test(6:2:2), 2:k-folds'."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for data partition."
    )
    parser.add_argument(
        "--train_method",
        type=str,
        default="baseline",
        help="Training method for the model. (should be specified in the configs/train_configs.yaml file)"
    )
    parser.add_argument(
        "--reweight",
        action='store_true',
        help="Sample a balanced dataset."
    )
    parser.add_argument(
        "--reweight_cols",
        nargs='+',
        choices=['label'],
        default=['label'],
        help="Columns to reweight the dataset. Default is ['label'].")
    
    parser.add_argument(
        "--reweight_method",
        type=str,
        choices=['none', 'undersample', 'oversample', 'weightedsampler'],
        default='undersample',
        help="Method for reweighting the dataset.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, don't re-train or re-infer any gene for which output already exists."
    )
    parser.add_argument(
        "--pretrained",
        action='store_true',
        help="Use pretrained weights.")

    parser.add_argument(
        "--finetune_layer_names",
        nargs='+',
        default=None,
        help="Layers to finetune. If not specified, all layers are finetuned.")
    parser.add_argument(
        "--class_loss",
        type=str,
        default='CrossEntropy',
        choices=['CrossEntropy', 'QALY'],
        help="Loss function for classification")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--acc_grad",
        type=int,
        default=1,
        help="Accumulation gradient."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate for the model"
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=1,
        help="Gamma for scheduler"
    )
    parser.add_argument(
        "--scheduler_step",
        type=float,
        default=10,
        help="Steps for scheduler"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=1.0,
        help="Split ratio for training set"
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default='TCGA',
        choices=['TCGA', 'CPTAC','DFCI', 'RoswellPark'],
        help="embeddings data source used to train the model. If not specified, --feature_paths must be provided"
    )
    parser.add_argument(
        "--embeddings_base_path",
        type=str,
        default='/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/',
        help="base path for the embeddings"
    )

    parser.add_argument(
        "--feature_paths",
        type=str,
        nargs='+',
        default=None,
        help="feature paths for different data sources. Can be a list of paths"
    )
    parser.add_argument(
        "--slide_type",
        type=str,
        default='PM',
        choices=['PM', 'FS','mixed'],
        help="type of slide, either PM or FS"
    )
    parser.add_argument(
        "--cutoff_method",
        type=str,
        choices=list(typing.get_args(CUTOFF_METHODS)),
        default='none',
        help="Cut off method for binary classification")

    parser.add_argument(
        "--input_feature_length",
        type=int,
        default=768,
        help="input feature length of different foundation model. 768 for CHIEF, 1024 for UNI, 1536 for GigaPath"
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="perform inference only for task 1, 2 and 3 in main.py"
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        nargs='+',
        default=['test'],
        choices=['valid', 'test','train','all'],
        help="Partition to perform inference on. Default is test."
    )
    parser.add_argument(
        "--sig_test_only",
        action="store_true",
        help="perform significance test only from preexisting results"
    )
    parser.add_argument(
        "--sig_n_bootstraps",
        type=int,
        default=1000,
        help="number of bootstraps for significance test"
    )
    
    parser.add_argument(
        "--magnification",
        type=int,
        default=20,
        help="Magnification of the patches in the training dataset"
    )
    parser.add_argument(
        "--stain_norm",
        action="store_true",
        help="indicate if the patches in the training dataset were stain normalised"
    )
    parser.add_argument(
        "--dataset_config_yaml",
        type=str,
        default="configs/dataset_configs.yaml",
        help="""yaml file for dataset configurations""")
    parser.add_argument(
        "--train_configs_yaml",
        type=str,
        default="configs/train_configs.yaml",
        help="""yaml file for training configurations""")
    parser.add_argument(
        "--label_map_yaml",
        type=str,
        default="configs/task_label_maps.yaml",
        help="""yaml file for mapping task labels""")
    parser.add_argument(
        "--show_testing_when_training",
        action="store_true",
        default=False,
        help="show testing results when training"
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        default=None,
        choices=[None, "bottleneck", "parallel"],
        help="Type of adapter to use. None for no adapter."
    )
    parser.add_argument(
        "--adapter_dim",
        type=int,
        default=None,
        help="Dimension of adapter bottleneck."
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    # check if args.task can be converted to int. If not, it is a custom task
    try:
        args.task = int(args.task)
    except:
        pass
    ## override args.feature_type based on the foundation model
    args.feature_type = FOUNDATION_MDL_FEATURE_TYPE_MAP[args.foundation_model]
    # Override the default training configs with the specified train method
    args = fetch_train_configs(args)
    ##
    if args.eval_batch_size is None:
        print("Eval batch size is not specified. Using the same as batch size.")
        args.eval_batch_size = args.batch_size
    if args.inference_output_path is None:
        print(f"Inference output path is not specified. Using the same as model_path.")
        args.inference_output_path = args.model_path
    ## if reweight is false, set reweight_method to none
    if not args.reweight:
        print("Reweight is false. Setting reweight_method to none.")
        args.reweight_method = 'none'
    if print:
        print("=========\tArguments\t=========")
        for arg in vars(args):
            print(f"\t{arg}:\t{getattr(args, arg)}")
        print("========\tEnd of Arguments\t=======")
    return args

def fetch_train_configs(args):
    # Override the default training configs with the specified train method
    
    # load the algorithm yaml config file
    with open(args.train_configs_yaml) as file:
        train_configs = yaml.load(file, Loader=yaml.FullLoader)
    # check if the train method is in the yaml file
    assert args.train_method in train_configs, ValueError(
            f"train method {args.train_method} not found in the training config file")
    # add the train configs to the args
    train_args = train_configs[args.train_method]
    print(f"Adding train configs for <<{args.train_method}>> to the args:")
    for arg in train_args:
        print(f"\t{arg}:\t{train_args[arg]}")
    args.__dict__.update(train_args)
    return args
    
def save_args(args, dir_to_save, filename='args.json'):
    # Convert Namespace to dictionary
    args_dict = vars(args)
    file_path = os.path.join(dir_to_save, filename)
    # Export args_dict to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    print(f"Arguments have been saved to {file_path}")

def generate_dataset(args):
    ## load dataset configuration
    with open(args.dataset_config_yaml, 'r') as stream:
        try:
            dataset_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    dataset_config = dataset_configs[args.data_source]
    ##
    strEmbeddingPath = construct_wsi_embedding_path(args)
    if 'geneType' not in args:
        args.geneType = ''
    if 'geneName' not in args:
        args.geneName = ''
    assert args.data_source in dataset_configs.keys(), f"Dataset {args.data_source} not found in the dataset configuration yaml ({args.dataset_config_yaml})."
    ## get dataset generator class from string
    generator_name = dataset_config['generator']
    dataset_generator = globals()[generator_name]
    ## define default dataset arguments
    intDiagnosticSlide_mapping = {"PM": 1, "FS": 0, "mixed": None}
    dataset_args = {
        'cancer': args.cancer,
        'fold': args.partition,
        'task': args.task,
        'seed': args.seed,
        'feature_type': args.feature_type,
        'strEmbeddingPath': strEmbeddingPath,
        'intDiagnosticSlide': intDiagnosticSlide_mapping[args.slide_type],
        'label_map_yaml': args.label_map_yaml,
        'geneType': args.geneType,
        'geneName': args.geneName,
    }
    ## get dataset-specific arguments
    dataset_new_args = dataset_config['args']
    ## update dataset-specific arguments with the provided arguments
    print(f"Overriding dataset-specific arguments with the provided arguments:")
    for k, v in dataset_new_args.items():
        print(f" \t{k}:\t{v}")
    dataset_args.update(dataset_new_args)
    ## instantiate the dataset generator
    data = dataset_generator(**dataset_args)
    
    return data

def get_model(args, num_classes):
    if args.feature_type == 'tile':
        if args.task in [1, 2, 4] or type(args.task) == str:
            print(f"Using ClfNet for task {args.task}")
            if args.task == 4:  # gene mutation
                model = ClfNet(
                    featureLength=args.input_feature_length,
                    classes=num_classes,
                    dropout=args.dropout,
                    adapter_type=args.adapter_type,
                    adapter_dim=args.adapter_dim,
                )
            else:
                # if no adapter needed for the other tasks
                model = ClfNet(
                    featureLength=args.input_feature_length,
                    classes=num_classes,
                    dropout=args.dropout,
                    adapter_type=args.adapter_type,
                    adapter_dim=args.adapter_dim,
                )
        elif args.task == 3:
            print(f"Using WeibullModel for task {args.task}")
            model = WeibullModel(featureLength=args.input_feature_length, dropout=args.dropout)
    return model


def construct_wsi_embedding_path(args):
    if args.foundation_model == 'TITAN':
        if args.data_source == 'TCGA':
            cancer_to_path = '/n/data2/hms/dbmi/kyu/lab/NCKU/TCGA_TITAN_features.pkl'
        else:
            raise notImplementedError(f"Data source {args.data_source} not supported for TITAN model.")
        return cancer_to_path
        
    if args.stain_norm:
        stain_norm_str = "(stain_norm)"
    else:
        stain_norm_str = ""
    # base = "/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/"
    mag = f"{args.magnification}X"
    cancer_to_path = {}

    for cancer in args.cancer:
        cancer = cancer.upper()
        if cancer == "COADREAD":  # handle task4 naming convention
            cancer_to_path["COAD"] = _get_feature_path(
                args, "COAD", mag, stain_norm_str)
            cancer_to_path["READ"] = _get_feature_path(
                args, "READ", mag, stain_norm_str)
            continue
        cancer_to_path[cancer] = _get_feature_path(
            args, cancer, mag, stain_norm_str)

    return cancer_to_path

def _get_feature_path(args, cancer, mag, stain_norm_str):
    base = args.embeddings_base_path
    if args.data_source == "TCGA":
        if args.slide_type == 'mixed':
            return [join(base, f"TCGA-{cancer}-PM/{args.foundation_model}/{mag}/pt_files{stain_norm_str}"), \
                join(base, f"TCGA-{cancer}-FS/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")]
        return join(base, f"TCGA-{cancer}-{args.slide_type}/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")
    else:
        raise NotImplementedError(f"Data source {args.data_source} not supported.")

def train_os_settings(args):
    max_index = None
    reweight_str = f"_finetune_{args.train_method}" if args.pretrained else ""
    cancer_folder = f'{args.task}_{"_".join(args.cancer)}'
    dir = join(args.model_path,
               f"{cancer_folder}_{args.partition}{reweight_str}/")
    os.makedirs(dir, exist_ok=True)

    folder_names = os.listdir(dir)
    subfolders = [folder for folder in folder_names if os.path.isdir(
        os.path.join(dir, folder))]
    if not subfolders:
        max_index = 1
    else:
        model_indexes = [int(name.split('_')[0]) for name in subfolders]
        max_index = max(model_indexes)
        if args.partition == 1:
            max_index += 1
        if args.partition == 2 and args.curr_fold == 0:
            max_index += 1

    return max_index, reweight_str

def test_os_settings(args):
    if args.task != 4:
        max_index = None
        reweight_str = f"_finetune_{args.train_method}" if args.pretrained else ""
        cancer_folder = f'{args.task}_{"_".join(args.cancer)}'
        dir = join(args.model_path,
                f"{cancer_folder}_{args.partition}{reweight_str}/")
        os.makedirs(dir, exist_ok=True)

        folder_names = os.listdir(dir)
        subfolders = [folder for folder in folder_names if os.path.isdir(
            os.path.join(dir, folder))]
        assert len(subfolders) > 0, f"No model folders found in {dir}"
        model_indexes = [int(name.split('_')[0]) for name in subfolders]
        max_index = max(model_indexes)
        # search the subfolders with the max index
        subfolders = [folder for folder in subfolders if folder.startswith(str(max_index))]
        maxidx_folders = []
        for fold in range(N_FOLDS):
            fold_folders = [folder for folder in subfolders if folder.endswith(f"_{fold}")]
            assert len(fold_folders) > 0, f"No model folders found for fold {fold} in {dir}"
            maxidx_folders.append(join(dir, fold_folders[0]))
        return max_index, maxidx_folders
    else:
        results_path = join(args.model_path, args.cancer[0].lower())
        
        for models in os.listdir(results_path):
            # the folders should be in the format of task_cancer_geneType_geneName_freq_partition
            # e.g. 4_brca_Common Genes_CDH1-Percentage_12.2_2
            if len(models.split("_")) == 6:

                geneType = models.split("_")[2]
                geneName = "_".join(models.split("_")[3:-1])
                geneName_short = geneName.split('-')[0]
                if args.gene not in geneName:
                    continue
                reweight_str = f"_finetune_{args.train_method}" if args.pretrained else ""
                cancer_folder = f"{args.task}_{ '_'.join(args.cancer)}_{geneType}_{geneName}_{args.partition}"
                gene_weight_folder = case_insensitive_glob(join(results_path, f"{cancer_folder}{reweight_str}"))[0]
                model_names = os.listdir(gene_weight_folder)
                subfolders = [folder for folder in model_names if os.path.isdir(
                    os.path.join(gene_weight_folder, folder))]
                model_indexes = [int(name.split('_')[0])
                                    for name in subfolders]
                max_index = max(model_indexes)
                # search the subfolders with the max index
                subfolders = [folder for folder in subfolders if folder.startswith(str(max_index))]
                
                maxidx_folders = []
                for fold in range(N_FOLDS):
                    fold_folders = [folder for folder in subfolders if folder.replace(reweight_str,'').endswith(f"_{fold}")]
                    assert len(fold_folders) > 0, f"No model folders found for fold {fold} in {gene_weight_folder}"
                    maxidx_folders.append(join(gene_weight_folder, fold_folders[0]))
                return max_index, maxidx_folders

def loss_fn_settings(args, ds):
    # get the loss function for the task
    if args.reweight_method == 'weightedsampler': 
        # if using weighted sampler, set the weight to None
        weight = None
    elif args.task != 3:
        label_counts = ds.df['label'].value_counts(
        ).sort_index().to_numpy()
        label_weight = 1 / label_counts
        label_weight = label_weight / label_weight.sum()
        # label_counts =  label_counts /
        weight = torch.tensor(label_weight).to(args.device)
    else:
        weight = None
    if args.class_loss == 'CrossEntropy':
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Loss function {args.class_loss} not supported.")

    return loss_fn

def task_collate_fn_settings(args):
    # get the collate function for the task
    if args.feature_type == 'tile':
        if args.task == 3:  # survival prediction
            return survival_collate_fn
        else:
            return collate_fn
    elif args.feature_type == 'slide':
        if args.task == 3:
            raise NotImplementedError("Survival prediction on slide level is not implemented.")
        else:
            return slide_level_collate_fn

def train_mutation_os_settings(args):
    reweight_str = f"_finetune_{args.train_method}" if args.pretrained else ""
    directory_path = f"./tcga_pan_cancer/{args.cancer[0]}_tcga_pan_can_atlas_2018/"
    if os.path.isdir(directory_path):
        for types in os.listdir(directory_path):
            if types == 'Common Genes':
                geneType = 'Common Genes'
            elif types == 'Targeted Drugs for Genes':
                geneType = 'Mutated Genes'

            if os.path.isdir(f'{directory_path}/{types}/'):
                for gName in os.listdir(f"{directory_path}/{types}/"):
                    geneName = "_".join(gName.split('_')[1:])
                    cancer_folder = f'{args.task}_{"_".join(args.cancer)}_{geneType}_{geneName}'
                    dir = join(
                        args.model_path, f"{cancer_folder}_{args.partition}{reweight_str}/")
                    os.makedirs(dir, exist_ok=True)

                    folder_names = os.listdir(dir)
                    subfolders = [folder for folder in folder_names if os.path.isdir(
                        os.path.join(dir, folder))]
                    if not subfolders:
                        max_index = 1
                    else:
                        model_indexes = [int(name.split('_')[0])
                                         for name in subfolders]
                        max_index = max(model_indexes)
                        if args.partition == 1:
                            max_index += 1
                        if args.partition == 2 and args.curr_fold == 0:
                            max_index += 1

    return max_index, reweight_str

def wandb_setup(args, max_index, reweight_str, gene=None):

    cancer_part = "_".join(args.cancer)
    fold_part = f"_fold_{args.curr_fold}" if args.partition == 2 else ""
    gene_part = f"_{gene}" if gene else ""
    job = f"{max_index}_{cancer_part}{gene_part}{fold_part}{reweight_str}"
    task = f"{cancer_part}_{gene_part}{str(args.partition)}{reweight_str}"
    wandb_group = f"{args.task}_{task}"

    wandb.init(project=f'Gene-Mutation-{args.train_method}',
               config={"_service_wait": 6000, **vars(args)},
               name=job,
               group=wandb_group,
               tags=["loss_finetuning"])

    return job, task

def load_pretrained_weights(args, num_classes, max_index):
    model = get_model(args,num_classes)

    if args.task != 4: # for all tasks other than mutation prediction
        cancer_folder = f'{args.task}_{"_".join(args.cancer)}_{args.partition}'
    else:
        cancer_folder = join(
            args.cancer[0],  
            f'{args.task}_{"_".join(args.cancer)}_{args.geneType}_{args.geneName}_{args.partition}'
        )

    if args.weight_path != "":
        weight_path_sstr = join(
            args.model_path, cancer_folder, f"{args.weight_path}_*/model.pt")
    else:
        if args.partition == 1:
            weight_path_sstr = join(
                args.model_path, cancer_folder, f"{max_index}_*/model.pt")

        elif args.partition == 2:
            weight_path_sstr = join(
                args.model_path, cancer_folder, f"{max_index}_*_{args.curr_fold}", "model.pt")
    
    weight_paths = glob.glob(weight_path_sstr)
    if len(weight_paths) == 0:
        print(f"Warning: Weight path not found: {weight_path_sstr}. Skipping this model.")
        return None  # Return None if no pretrained file is found

    weight_path = weight_paths[0]
    model.load_state_dict(torch.load(
        weight_path, map_location=args.device), strict=False)
    print(f"Weights path:{weight_path}")
    print("Loaded pretrained weights.")
    return model

def optimizer_settings(args, model):
    parameters_to_update = []
    parameter_names_to_update = []
    n_params = 0

    if not args.pretrained:  # if not pretrained, enable all layers
        for n, p in model.named_parameters():
            p.requires_grad = True
            parameters_to_update.append(p)
            parameter_names_to_update.append(n)
            n_params = n_params + np.prod(p.size())
    else:  # if pretrained, only enable the specified layers in args.finetune_layer_names
        finetune_layer_list = args.finetune_layer_names
        for n, p in model.named_parameters():
            # if any([n.startswith(layer) for layer in finetune_layer_list]):
            if finetune_layer_list == None or any([n.startswith(layer) for layer in finetune_layer_list]):

                p.requires_grad = True
                parameters_to_update.append(p)
                parameter_names_to_update.append(n)
                # get total number of elements in the torch tensor
                n_params = n_params + np.prod(p.size())
            else:
                p.requires_grad = False

    if args.task == 1 or args.task == 2 or args.task == 4 or type(args.task) == str:
        optimizer = torch.optim.Adam(parameters_to_update, lr=args.lr)
    elif args.task == 3:
        optimizer = torch.optim.RMSprop(
            parameters_to_update, lr=args.lr, weight_decay=1e-4 if args.pretrained else 0)

    print(f"Params to learn:{n_params}")
    # [print(f'\t{n}') for n in parameter_names_to_update]
    return optimizer


def run(args, train_eval_dl, model, num_classes, colour, loss_fn, optimizer=None, epoch=None):
    total_train_eval_loss = 0.
    logits = []
    probs = []
    predictions = []
    predicted_survival_times = []
    true_survival_times = []
    labels = []
    events = []
    caseIds = []
    slideIds = []
    pbar = tqdm(enumerate(train_eval_dl), colour=colour,
                total=len(train_eval_dl), mininterval=10)

    if args.task == 3:
        loss_fn_wrapped = GroupStratifiedSurvivalLoss()
    else:
        loss_fn_wrapped = GroupStratifiedLoss(loss_fn)

    for idx, data in pbar:

        if args.task == 3:
            wsi_embeddings, lengths, event, time, group, stage, case_id, slide_id = data
            shape_scale = model(wsi_embeddings.to(args.device), lengths)
            shape, scale = shape_scale[:, 0], shape_scale[:, 1]
            label = None
            train_eval_loss, group_of_loss = loss_fn_wrapped(shape, scale, time.float().to(
                args.device), torch.nn.functional.one_hot(event, num_classes).float().to(args.device), lengths, group)
        else:
            wsi_embeddings, lengths, label, group, case_id, slide_id = data
            cancer_pred = model(wsi_embeddings.to(args.device), lengths)
            train_eval_loss, group_of_loss = loss_fn_wrapped(
                cancer_pred, torch.nn.functional.one_hot(label, num_classes).float().to(args.device), group)
        train_eval_loss = train_eval_loss / args.acc_grad

        if model.training:
            train_eval_loss.backward()
            if (idx + 1) % args.acc_grad == 0:
                optimizer.step()
                optimizer.zero_grad()

        if not torch.isnan(train_eval_loss):
            total_train_eval_loss += train_eval_loss.detach().cpu().numpy()

        avg_train_eval_loss = total_train_eval_loss / (idx+1)

        pbar.set_description(
            (f"Iter:{epoch+1:3}/{args.epochs:3} " if epoch is not None else "") +
            f"Avg_loss:{avg_train_eval_loss:.4f}", refresh=False)

        if not model.training:
            if args.task == 3:
                predicted_survival_time = scale * \
                    torch.exp(torch.log(time.to(args.device) + 1e-8) / shape)
                predicted_survival_times.append(
                    predicted_survival_time.detach().cpu().numpy())
                true_survival_times.append(time.detach().cpu().numpy())
                events.append(event.detach().cpu().numpy())
            else:
                predictions.append(torch.argmax(
                    cancer_pred.detach().cpu(), dim=1).numpy())

                logits.append(cancer_pred.detach().cpu().numpy())
                probs.append(torch.nn.functional.softmax(
                    cancer_pred, dim=1).detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

            caseIds.append(case_id)
            slideIds.append(slide_id)

    if model.training:
        return avg_train_eval_loss
    else:
        if args.task == 3:
            true_survival_times = np.concatenate(true_survival_times, axis=0)
            predicted_survival_times = np.concatenate(
                predicted_survival_times, axis=0)
            events = np.concatenate(events, axis=0)
        else:
            predictions = np.concatenate(predictions, axis=0)
            logits = np.concatenate(logits, axis=0)
            probs = np.concatenate(probs, axis=0)
            labels = np.concatenate(labels, axis=0)
        caseIds = [item for sublist in caseIds for item in sublist]
        slideIds = [item for sublist in slideIds for item in sublist]
        return (labels, events, true_survival_times, predicted_survival_times, predictions, probs, logits, caseIds, slideIds), avg_train_eval_loss

def test_folder_setup(args, curr_fold,inference_mode:Literal['valid', 'test','train','all']='test'):
    inf_mode_prefix_map = {'valid': 'valid_', 'test': '', 'train': 'train_', 'all': 'all_'}
    inf_mode_prefix = inf_mode_prefix_map[inference_mode]

    reweight_str = f"_finetune_{args.train_method}" if args.pretrained else ""
    fold_str = f"_fold_{curr_fold}" if curr_fold is not None and args.partition == 2 else ""

    cancer_folder = f'{args.task}_{"_".join(args.cancer)}'

    dir_path = join(args.model_path,
                    f"{cancer_folder}_{args.partition}{reweight_str}")

    model_names = os.listdir(dir_path)
    subfolders = [folder for folder in model_names if os.path.isdir(
        os.path.join(dir_path, folder))]
    # get the model index:
    # the first field is the run ID (e.g. for 2-base-kmsjtt4p_fold_3, 2 is the run ID)
    model_indexes = [int(name.split('_')[0]) for name in subfolders]
    # The run ID increases by 1 for each new run. The max run ID is the most recent run.
    max_index = max(model_indexes) if args.weight_path == "" else int(
        args.weight_path)

    # weight_path = glob.glob(join(dir_path,f"{max_index}_*{fold_str}{reweight_str}/model.pt"))[0]
    weight_path_sstr = join(
        dir_path, f"{max_index}_*{fold_str}{reweight_str}/model.pt")
    weight_paths = glob.glob(weight_path_sstr)
    assert len(weight_paths) > 0, f"Weight path not found: {weight_path_sstr}"
    weight_path = weight_paths[0]

    parent_weight_path = Path(weight_path).parent if args.partition == 1 else Path(
        weight_path).parent.parent
    if args.inference_output_path is None or args.inference_output_path == args.model_path:
        result_path = parent_weight_path / f"{max_index}_result.csv"
        model_names = os.listdir(dir_path)
        fig_path = parent_weight_path / \
            f"{str(max_index) + '_' if args.partition == 2 else ''}survival_curve.png"
        fig_path2 = parent_weight_path / \
            f"{str(max_index) + '_' if args.partition == 2 else ''}survival_curve_stage.png"
        fig_path3 = parent_weight_path / \
            f"{str(max_index) + '_' if args.partition == 2 else ''}survival_curve_black.png"
        kfold_results_path = Path(weight_path).parent  / f"{inf_mode_prefix}inference_results_fold{curr_fold}.csv" if args.partition == 2 else ""
        parent_result_path = Path(result_path).parent
    else:
        cancer_part = "_".join(args.cancer)
        fold_part = f"_fold_{args.curr_fold}" if args.partition == 2 else ""

        outdir_path = join(args.inference_output_path,
                        f"{cancer_folder}_{args.partition}{reweight_str}")
        result_path = join(outdir_path, f"{max_index}_result.csv")
        parent_result_path = Path(result_path).parent
    
        fig_path = parent_result_path / \
            f"{str(max_index) + '_' if args.partition == 2 else ''}survival_curve.png"
        fig_path2 = parent_result_path / \
            f"{str(max_index) + '_' if args.partition == 2 else ''}survival_curve_stage.png"
        fig_path3 = parent_result_path / \
            f"{str(max_index) + '_' if args.partition == 2 else ''}survival_curve_black.png"
        kfold_results_path = parent_result_path / f"{inf_mode_prefix}inference_results_fold{curr_fold}.csv" if args.partition == 2 else ""
    os.makedirs(parent_result_path, exist_ok=True)
    

    return result_path, fig_path, fig_path2, fig_path3, weight_path, kfold_results_path


def test_load_pretrained_weights(args, weight_path, num_classes):
    model = get_model(args,num_classes)

    try:
        model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=False)
    except FileNotFoundError:
        print(f"Warning: Weight path not found: {weight_path}. Skipping this model.")
        return None  # Return None if no pretrained file is found

    return model


def test_save_results(args, num_classes, labels, predictions, probs, result_path):
    if num_classes > 2:
        results = {"accuracy": (predictions == labels).mean()}
        pd.DataFrame(results).T.to_csv(result_path)
        print(f"Save results to:{result_path}")

    elif num_classes == 2:
        auroc = roc_auc_score(labels, probs)
        results = {"AUROC": auroc, "accuracy": (predictions == labels).mean()}
        pd.DataFrame(results).T.to_csv(result_path)
        print(f"Save results to:{result_path}")

    print(pd.DataFrame(results).T)


def test_save_kfold_results(logits, probs, predictions, labels, caseIds, slideIds, kfold_results_path, survival_res=None):
    if not survival_res:
        inference_results = pd.DataFrame({
            "logits": logits,
            "prob": probs,
            "pred": predictions,
            "label": labels,
            "patient_id": caseIds,
            "slide_id": slideIds,
        })
        inference_results["pred"] = inference_results["pred"].astype(int)
        inference_results["label"] = inference_results["label"].astype(int)
    else:
        inference_results = pd.DataFrame({
            "predicted_survival_times": survival_res[0],
            "true_survival_times": survival_res[1],
            "stages": survival_res[2],
            "events": survival_res[3],
            "patient_id": caseIds,
            "slide_id": slideIds,
        })
        inference_results["predicted_survival_times"] = inference_results["predicted_survival_times"].astype(
            float)
        inference_results["true_survival_times"] = inference_results["true_survival_times"].astype(
            float)
        inference_results["stages"] = inference_results["stages"].astype(int)
        inference_results["events"] = inference_results["events"].astype(int)

    inference_results.to_csv(kfold_results_path)


def test_run(args, test_pbar, model, weight_path=None):
    caseIds = []
    slideIds = []
    logits = []
    probs = []
    predictions = []
    labels = []
    events = []
    predicted_survival_times = []
    true_survival_times = []
    stages = []

    for _, data in test_pbar:
        if args.task != 3:
            wsi_embeddings, lengths, label, group, case_id, slide_id = data
            test_cancer_pred = model(wsi_embeddings.to(args.device), lengths)
            predictions.append(torch.argmax(
                test_cancer_pred.detach().cpu(), dim=1).numpy())
            labels.append(label.detach().cpu().numpy())

            caseIds.append(case_id)
            slideIds.append(slide_id)
            logits.append(test_cancer_pred.detach().cpu().numpy())
            probs.append(torch.nn.functional.softmax(
                test_cancer_pred, dim=1).detach().cpu().numpy())

        else:
            wsi_embeddings, lengths, event, time, group, stage, case_id, slide_id = data
            test_shape_scale = model(wsi_embeddings.to(args.device), lengths)
            test_shape, test_scale = test_shape_scale[:,
                                                      0], test_shape_scale[:, 1]

            predicted_survival_time = test_scale * \
                torch.exp(torch.log(time.to(args.device) + 1e-8) / test_shape)

            predicted_survival_times.append(
                predicted_survival_time.detach().cpu().numpy())
            true_survival_times.append(time.detach().cpu().numpy())
            events.append(event.detach().cpu().numpy())
            stages.append(stage.detach().cpu().numpy())
            caseIds.append(case_id)
            slideIds.append(slide_id)
    # convert to numpy arrays
    if args.task == 3:
        predicted_survival_time = np.concatenate(
            predicted_survival_times, axis=0)
        true_survival_time = np.concatenate(true_survival_times, axis=0)
        events = np.concatenate(events, axis=0)
        stages = np.concatenate(stages, axis=0)
    else:
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)
        probs = np.concatenate(probs, axis=0)
        probs = probs[:, 1]
        logits = logits[:, 1]

        inpath = dirname(weight_path)
        predictions = get_predictions(probs, inpath, method=args.cutoff_method)
    caseIds = [item for sublist in caseIds for item in sublist]
    slideIds = [item for sublist in slideIds for item in sublist]
    return slideIds, caseIds, logits, probs, predictions, labels, events, predicted_survival_times, true_survival_times, stages


                
