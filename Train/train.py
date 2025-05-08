import torch
from tqdm import tqdm
import numpy as np
from utils import flatten_batches

def run_epoch(model, loader, loss_fn, optimizer, device, train=True):
    mode = "train" if train else "eval"
    if train: model.train()
    else:     model.eval()

    total_loss = 0.0
    all_logits, all_probs, all_preds, all_labels, all_sens = [], [], [], [], []

    pbar = tqdm(loader, desc=f"{mode} pass")
    for batch in pbar:
        embeddings, lengths, sensitive, label, *_ = batch
        embeddings = embeddings.to(device); sensitive = sensitive.to(device)
        label = label.to(device)

        logits = model(embeddings, sensitive, lengths)
        loss = loss_fn(logits, label)

        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=1)[:,1]
            preds = (probs > 0.5).long()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu().numpy())
        all_probs .append(probs .detach().cpu().numpy())
        all_preds .append(preds .detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())
        all_sens  .append(sensitive.detach().cpu().numpy())

        pbar.set_postfix(loss=total_loss/(len(all_labels)))

    return {
        "loss": total_loss/len(loader),
        "logits": np.concatenate(all_logits),
        "probs" : np.concatenate(all_probs),
        "preds" : np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
        "sens"  : np.concatenate(all_sens),
    }

def optimizer_settings(args, model):
    return torch.optim.Adam(model.parameters(), lr=args.lr)

def loss_fn_settings(args, ds):
    # simple cross‐entropy (ignore reweight for now)
    return torch.nn.CrossEntropyLoss()
