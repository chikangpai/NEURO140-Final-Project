import os
import datetime
from pathlib import Path
from typing import Literal, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from framework import parse_args

from dataset import generate_dataset, get_datasets, CancerDataset, N_FOLDS  # noqa: F401  (external module)
from network import ClfNet  # noqa: F401 (external module)
from util import (
    save_args,
    optimizer_settings,
    loss_fn_settings,
    get_model,
    case_insensitive_glob,
)  # noqa: F401 (external module)

# ---------------------------------------------------------------------------
# CONSTANTS & HELPERS
# ---------------------------------------------------------------------------
TASK_GENE_MUTATION: int = 3       # Fixed task id for gene‑mutation classification
NUM_CLASSES: int = 2             # Gene‑mutation is binary (mutated vs wild‑type)

# ---------------------------------------------------------------------------
# DATA HELPERS (GENE LIST)
# ---------------------------------------------------------------------------

def list_available_genes(args, use_abbr: bool = False) -> List[str]:
    """Return available gene names for the requested cancer type.

    This is an extremely reduced variant of the original helper that *only*
    supports TCGA pan‑cancer data. All other branches and commentary have
    been removed to keep the function focused on gene‑mutation training.
    """
    if args.data_source != "TCGA":
        raise NotImplementedError("Only TCGA source supported in the refactor.")

        directory_path = f"./tcga_pan_cancer/{args.cancer[0].lower()}_tcga_pan_can_atlas_2018"
    if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

    gene_list: List[str] = []
    for sub in os.listdir(directory_path):
        if not os.path.isdir(os.path.join(directory_path, sub)):
            continue
        for g_name in os.listdir(os.path.join(directory_path, sub)):
            if g_name.endswith("_"):
                g_name = g_name[:-1]  # trim trailing underscore if present
            full_gene = "_".join(g_name.split("_")[1:])
            gene_list.append(full_gene.split("-")[0] if use_abbr else full_gene)
    return sorted(set(gene_list))


# ---------------------------------------------------------------------------
# TRAINING / VALIDATION LOOP (SINGLE FOLD)
# ---------------------------------------------------------------------------

def train_validate_single_fold(args, df, fold_id: int):
    """Train + validate on a single fold. Returns best AUROC obtained."""
    train_ds, val_ds, _ = get_datasets(
        df,
        TASK_GENE_MUTATION,
        split_type="kfold" if args.partition == 2 else "vanilla",
        curr_fold=fold_id,
        feature_type=args.feature_type,
    )

    # DataLoaders – *no* sensitive attributes needed any more, so the default
    # collate_fn works fine.
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.n_workers, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.n_workers, pin_memory=False)

    # ------------------------------------------------------------------
    # MODEL, OPTIMISER, LOSS
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    model = get_model(args, NUM_CLASSES).to(args.device)
    optimiser = optimizer_settings(args, model)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )
    loss_fn = loss_fn_settings(args, train_ds)

    best_auroc: float = 0.0
    epochs_without_improvement: int = 0

    # ------------------------------------------------------------------
    # EPOCH LOOP
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        # ------------------ TRAIN ------------------
        model.train()
        train_loss_accum = 0.0
        for x, label, *_ in tqdm(train_dl, desc=f"Train Fold {fold_id} Epoch {epoch}"):
            optimiser.zero_grad()
            logits = model(x.to(args.device))
            loss = loss_fn(logits, label.to(args.device))
            loss.backward()
            optimiser.step()
            train_loss_accum += loss.item() * x.size(0)
        scheduler.step()
        avg_train_loss = train_loss_accum / len(train_ds)

        # ------------------ VALID ------------------
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, label, *_ in DataLoader(val_ds, batch_size=args.eval_batch_size,
                                         shuffle=False, num_workers=args.n_workers):
                logits = model(x.to(args.device))
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                all_probs.append(probs)
                all_labels.append(label.numpy())
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        auroc = roc_auc_score(all_labels, all_probs)

        # --------------- WANDB LOG -----------------
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_auroc": auroc,
        })

        # --------------- EARLY STOPPING ------------
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), args.best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(f"Early‑stopped after {epoch} epochs – no AUROC gain.")
                break

    return best_auroc


# ---------------------------------------------------------------------------
# CROSS‑VALIDATION DRIVER (TRAIN + OPTIONAL INFERENCE)
# ---------------------------------------------------------------------------

def main_CV(args):
    """Replacement for main_CV that *only* handles gene‑mutation (task 3).

    ‑ Always forces `args.task = 3`.
    ‑ Removes survival‑, sensitive‑, and fairness‑related logic.
    """
    args.task = TASK_GENE_MUTATION

    # ------------------------------------------------------------------
    # Find genes to train
    # ------------------------------------------------------------------
    gene_list = list_available_genes(args)
    for gene in gene_list:
        if args.genes and not any(substr in gene for substr in args.genes):
            continue  # user provided a subset filter
        args.gene = gene

        # ----------------‑ DATASET -----------------
        data = generate_dataset(args)
        df = data.train_valid_test(args.split_ratio)

        # ----------------‑ CV FOLDS ---------------
        folds = range(1) if args.partition == 1 else range(N_FOLDS)
        best_fold_aurocs: List[float] = []
        for fold_id in folds:
            print(f"\n▶︎ Gene = {gene} · Fold = {fold_id}")

            # Set up a dedicated output directory and wandb run for the fold
            out_dir = Path(args.model_path) / gene / f"fold{fold_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            args.best_model_path = str(out_dir / "best_model.pt")
            wandb.init(project="GeneMutation", name=f"{gene}_fold{fold_id}", config=args)

            best_fold_aurocs.append(train_validate_single_fold(args, df, fold_id))
            wandb.finish()

        print(f"Finished {gene}. Mean AUROC across folds: {np.mean(best_fold_aurocs):.4f}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT (kept tiny – parse_args assumed external)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("Start:", datetime.datetime.now())
    args = parse_args()
    main_CV()
