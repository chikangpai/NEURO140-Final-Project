from torch.utils.data import DataLoader
from framework import generate_dataset, get_datasets, task_collate_fn_settings

def make_tcga_mutation_datasets(args):
    # prepare the three splits for this cancer+gene
    args.partition = 1
    args.curr_fold = None
    data = generate_dataset(args)
    df = data.train_valid_test(args.split_ratio)
    train_ds, val_ds, test_ds = get_datasets(
        df, args.task, "vanilla", None,
        feature_type=args.feature_type,
        reweight_method=args.reweight_method,
        reweight_cols=args.reweight_cols,
        max_train_tiles=args.max_train_tiles
    )
    collate = task_collate_fn_settings(args)
    return train_ds, val_ds, test_ds, collate
