"""
train_A.py  –  COL761 Assignment 3: Dataset A (Multi-class Node Classification)

Architecture : GATv2 + LayerNorm + Residual + DropEdge
Post-process : Pseudo-labeling (2 rounds) + gentle APPNP smoothing
Boosters     : 8-seed Ensemble, ReduceLROnPlateau, Early Stopping
Metric       : Accuracy
"""

import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch_geometric.nn import GATv2Conv
try:
    from torch_geometric.utils import dropout_edge
except ImportError:
    from torch_geometric.utils import dropout_adj
    def dropout_edge(edge_index, p=0.5, training=True):
        return dropout_adj(edge_index, p=p, training=training)

from load_dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class GATv2Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, heads=4, dropout=0.35):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        self.res_projs.append(
            nn.Linear(in_channels, hidden_channels * heads)
            if in_channels != hidden_channels * heads else nn.Identity()
        )

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads,
                          concat=True, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
            self.res_projs.append(nn.Identity())

        # Last layer
        self.convs.append(
            GATv2Conv(hidden_channels * heads, out_channels, heads=1,
                      concat=False, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(out_channels))
        self.res_projs.append(nn.Linear(hidden_channels * heads, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            residual = self.res_projs[i](x)
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Label Smoothing Loss
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_prob = F.log_softmax(logits, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  APPNP-style gentle smoothing via scipy
# ─────────────────────────────────────────────────────────────────────────────

def appnp_smooth(y_soft, data, alpha=0.1, n_iter=10):
    """
    Very gentle APPNP smoothing: Z = alpha*Y_soft + (1-alpha)*A_norm@Z
    Low alpha = mostly keep model predictions, tiny nudge from neighbors.
    """
    try:
        import scipy.sparse as sp
        N = data.num_nodes
        ei = data.edge_index
        row = np.concatenate([ei[0].numpy(), ei[1].numpy(), np.arange(N)])
        col = np.concatenate([ei[1].numpy(), ei[0].numpy(), np.arange(N)])
        vals = np.ones(len(row), dtype=np.float32)
        A = sp.csr_matrix((vals, (row, col)), shape=(N, N))
        deg = np.array(A.sum(axis=1)).flatten()
        D_inv = sp.diags(1.0 / np.maximum(deg, 1e-12))
        A_norm = D_inv @ A

        Z = y_soft.numpy().copy()
        Y0 = y_soft.numpy().copy()
        for _ in range(n_iter):
            Z = alpha * Y0 + (1.0 - alpha) * (A_norm @ Z)
        # Renormalize
        row_sums = Z.sum(axis=1, keepdims=True)
        Z = Z / np.maximum(row_sums, 1e-12)
        print(f"  APPNP smoothing done (alpha={alpha}, iters={n_iter})")
        return torch.FloatTensor(Z)
    except Exception as e:
        print(f"  APPNP smoothing failed: {e}")
        return y_soft


# ─────────────────────────────────────────────────────────────────────────────
#  Training Loop (single seed) — with DropEdge
# ─────────────────────────────────────────────────────────────────────────────

def train_one_seed(data, num_classes, in_channels, device, config, seed,
                   extra_train_idx=None, extra_train_labels=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = GATv2Model(
        in_channels=in_channels,
        hidden_channels=config['hidden'],
        out_channels=num_classes,
        num_layers=config['layers'],
        heads=config['heads'],
        dropout=config['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'], weight_decay=config['wd']
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.5,
                                  min_lr=1e-6)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    labeled_nodes = data.labeled_nodes.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)

    train_idx = labeled_nodes[train_mask]
    val_idx = labeled_nodes[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    # Augment with pseudo-labels if provided
    if extra_train_idx is not None and len(extra_train_idx) > 0:
        train_idx = torch.cat([train_idx, extra_train_idx.to(device)])
        y_train = torch.cat([y_train, extra_train_labels.to(device)])

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    early_stop_patience = config.get('early_stop', 50)
    drop_edge_p = config.get('drop_edge', 0.2)

    for epoch in range(1, config['epochs'] + 1):
        model.train()

        # DropEdge: randomly remove edges for regularization
        if drop_edge_p > 0:
            ei_drop, _ = dropout_edge(edge_index, p=drop_edge_p, training=True)
        else:
            ei_drop = edge_index

        optimizer.zero_grad()
        logits = model(x, ei_drop)
        loss = criterion(logits[train_idx], y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validate (full edges)
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            preds = logits[val_idx].argmax(dim=1)
            val_acc = (preds == y_val).float().mean().item()

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [Seed {seed}] Epoch {epoch:4d}  Loss={loss.item():.4f}  "
                  f"Val Acc={val_acc:.4f}  Best={best_val_acc:.4f}  LR={lr_now:.2e}")

        if patience_counter >= early_stop_patience:
            print(f"  [Seed {seed}] Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  [Seed {seed}] Best Val Accuracy = {best_val_acc:.4f}")
    return model, best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train model for Dataset A")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--drop_edge", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--pseudo_rounds", type=int, default=2)
    parser.add_argument("--pseudo_threshold", type=float, default=0.95)
    parser.add_argument("--smooth_alpha", type=float, default=0.1,
                        help="APPNP teleport probability (low = gentle)")
    parser.add_argument("--smooth_iters", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir = args.model_dir or args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading Dataset A ...")
    ds = load_dataset("A", args.data_dir)
    data = ds[0]
    num_classes = ds.num_classes
    in_channels = data.x.shape[1]
    print(f"  Nodes={data.num_nodes}  Edges={data.num_edges}  "
          f"Features={in_channels}  Classes={num_classes}")
    print(f"  Train={data.train_mask.sum().item()}  Val={data.val_mask.sum().item()}")

    # Feature normalization (row-normalize)
    row_norms = data.x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    data.x = data.x / row_norms

    config = {
        'hidden':     args.hidden,
        'layers':     args.layers,
        'heads':      args.heads,
        'lr':         args.lr,
        'wd':         args.wd,
        'dropout':    args.dropout,
        'epochs':     args.epochs,
        'early_stop': 50,
        'drop_edge':  args.drop_edge,
    }

    # ------------------------------------------------------------------
    # 2. Round 0: Train base ensemble (no pseudo-labels)
    # ------------------------------------------------------------------
    extra_idx = None
    extra_labels = None

    for rnd in range(args.pseudo_rounds + 1):
        if rnd == 0:
            print(f"\n{'='*60}")
            print(f"  Round 0: Training {len(args.seeds)}-seed base ensemble")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"  Round {rnd}: Pseudo-label retraining "
                  f"({len(extra_idx)} pseudo-labels added)")
            print(f"{'='*60}")

        all_soft_preds = []
        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            model, val_acc = train_one_seed(
                data, num_classes, in_channels, device, config, seed,
                extra_train_idx=extra_idx, extra_train_labels=extra_labels
            )

            model.eval()
            with torch.no_grad():
                logits = model(data.x.to(device), data.edge_index.to(device)).cpu()
            soft = torch.softmax(logits, dim=1)
            all_soft_preds.append(soft)

        # Average across seeds
        avg_soft = torch.stack(all_soft_preds).mean(dim=0)  # [N, C]

        # Evaluate
        labeled_nodes = data.labeled_nodes
        val_mask = data.val_mask
        val_idx = labeled_nodes[val_mask]
        y_val = data.y[val_mask]
        ensemble_preds = avg_soft[val_idx].argmax(dim=1)
        ensemble_acc = (ensemble_preds == y_val).float().mean().item()

        print(f"\n  Round {rnd} Ensemble Val Accuracy = {ensemble_acc:.4f}  ({ensemble_acc*100:.2f}%)")

        # Generate pseudo-labels for next round
        if rnd < args.pseudo_rounds:
            # Find unlabeled nodes
            all_labeled = set(data.labeled_nodes.tolist())
            all_nodes = set(range(data.num_nodes))
            unlabeled_nodes = torch.tensor(sorted(all_nodes - all_labeled), dtype=torch.long)

            if len(unlabeled_nodes) > 0:
                max_probs, pseudo_preds = avg_soft[unlabeled_nodes].max(dim=1)
                confident_mask = max_probs > args.pseudo_threshold

                extra_idx = unlabeled_nodes[confident_mask]
                extra_labels = pseudo_preds[confident_mask]
                print(f"  Pseudo-labeling: {confident_mask.sum().item()} nodes "
                      f"above threshold {args.pseudo_threshold}")
            else:
                extra_idx = torch.tensor([], dtype=torch.long)
                extra_labels = torch.tensor([], dtype=torch.long)

    # ------------------------------------------------------------------
    # 3. Apply gentle APPNP smoothing
    # ------------------------------------------------------------------
    if args.smooth_alpha > 0:
        print("\n  Applying APPNP smoothing ...")
        avg_soft = appnp_smooth(avg_soft, data,
                                alpha=args.smooth_alpha,
                                n_iter=args.smooth_iters)

    # Final evaluation
    ensemble_preds = avg_soft[val_idx].argmax(dim=1)
    final_acc = (ensemble_preds == y_val).float().mean().item()

    print(f"\n{'='*60}")
    print(f"  Final Ensemble Val Accuracy = {final_acc:.4f}  ({final_acc*100:.2f}%)")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 4. Save model
    # ------------------------------------------------------------------
    final_logits = torch.log(avg_soft + 1e-8)
    model_dict = {"logits": final_logits}

    model_path = os.path.join(out_dir, f"{args.kerberos}_model_A.pt")
    torch.save(model_dict, model_path)
    print(f"\nModel saved to {model_path}")
    print("Done!")


if __name__ == "__main__":
    main()
