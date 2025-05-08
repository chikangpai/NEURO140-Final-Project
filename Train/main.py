# Main file, learns a classifier for a given gene and cancer type
import datetime
import torch
import wandb

from args      import parse_args
from data      import make_tcga_mutation_datasets
from model     import build_model
from train     import run_epoch, optimizer_settings, loss_fn_settings
from utils     import save_args, roc_auc_score, get_predictions

def main():
    args = parse_args()
    print("Starting at", datetime.datetime.now())
    wandb.init(project="tcga‐mutation", name=f"{args.cancer}_{args.gene}")
    save_args(args, args.model_dir)

    # 1) data
    train_ds, val_ds, test_ds, collate = make_tcga_mutation_datasets(args)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate, shuffle=True,  num_workers=args.n_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.eval_batch_size, collate_fn=collate, shuffle=False, num_workers=args.n_workers)

    # 2) model + optim + loss
    torch.manual_seed(args.seed)
    model     = build_model(args, num_classes=2)
    optimizer = optimizer_settings(args, model)
    loss_fn   = loss_fn_settings(args, train_ds)

    # 3) train
    best_auc = 0.0
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, args.device, train=True)
        val_metrics   = run_epoch(model, val_loader,   loss_fn, None,      args.device, train=False)

        auc = roc_auc_score(val_metrics["labels"], val_metrics["probs"])
        print(f"Epoch {epoch:3d} | val AUROC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"{args.model_dir}/best.pt")
        wandb.log({f"train/loss":train_metrics["loss"], "val/loss":val_metrics["loss"], "val/AUROC":auc})

    # 4) test
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, collate_fn=collate, shuffle=False, num_workers=args.n_workers)
    test_metrics = run_epoch(model, test_loader, loss_fn, None, args.device, train=False)
    final_auc = roc_auc_score(test_metrics["labels"], test_metrics["probs"])
    preds = get_predictions(test_metrics["probs"])

    print("Test AUROC:", final_auc)

    wandb.finish()

if __name__ == "__main__":
    main()
