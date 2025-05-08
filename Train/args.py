import argparse
from framework import FOUNDATION_MDL_FEATURE_TYPE_MAP, fetch_train_configs

def parse_args():
    parser = argparse.ArgumentParser("TCGA gene‐mutation training")
    parser.add_argument("--cancer",        type=str,   required=True, help="TCGA cancer code, e.g. brca")
    parser.add_argument("--gene",          type=str,   required=True, help="Gene name for mutation task")
    parser.add_argument("--split_ratio",   type=float, default=0.6,   help="Train/val split fraction")
    parser.add_argument("--model_dir",     type=str,   required=True, help="Where to save models & logs")
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--n_workers",     type=int,   default=4)
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--dropout",       type=float, default=0.25)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--train_method",  type=str,   default="baseline")
    parser.add_argument("--reweight_method", type=str, choices=["none","undersample","oversample","weightedsampler"], default="undersample")
    parser.add_argument("--reweight_cols", nargs="+", default=["sensitive","label"])
    parser.add_argument("--max_train_tiles", type=int, default=None)
    parser.add_argument("--input_feature_length", type=int, default=768)
    parser.add_argument("--feature_type",  type=str, default="tile")
    parser.add_argument("--dataset_config_yaml", type=str, default="configs/dataset_configs.yaml")
    parser.add_argument("--train_configs_yaml",  type=str, default="configs/train_configs.yaml")
    parser.add_argument("--cutoff_method", type=str, default="none")
    parser.add_argument("--fair_attr",    type=str,   default=None, help="e.g. '{\"age\":[\"old\",\"young\"]}'")
    args = parser.parse_args()
    # map feature type if needed
    args.feature_type = FOUNDATION_MDL_FEATURE_TYPE_MAP[args.feature_type.upper()] \
                        if args.feature_type.upper() in FOUNDATION_MDL_FEATURE_TYPE_MAP \
                        else args.feature_type
    # apply any train‐method‐specific overrides
    args = fetch_train_configs(args)
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    return args
