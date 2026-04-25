"""
train_B.py  –  COL761 A3: Dataset B (Binary Node Classification)

Architecture : GraphSAGE GNN with mean aggregation
               Neighbor aggregation precomputed on CPU for scalability (2.89M nodes).
               Learnable SAGE transforms W_self and W_neigh applied on GPU.
Loss         : CrossEntropyLoss (pos_weight balanced)
Metric       : AUC-ROC
"""

import argparse, os, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from load_dataset import load_dataset


def precompute_neighbor_agg(data, chunk_size=100_000):
    """Compute A_norm @ X on CPU = mean-aggregated neighbor features (SAGE mean aggregator)."""
    import scipy.sparse as sp
    N = data.num_nodes
    src = data.edge_index[0].numpy().astype(np.int32)
    dst = data.edge_index[1].numpy().astype(np.int32)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    self_idx = np.arange(N, dtype=np.int32)
    row = np.concatenate([row, self_idx])
    col = np.concatenate([col, self_idx])
    vals = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((vals, (row, col)), shape=(N, N))
    deg = np.array(A.sum(axis=1)).flatten()
    D_inv = sp.diags(1.0 / np.maximum(deg, 1e-12))
    A_norm = (D_inv @ A).astype(np.float32)

    X = data.x.numpy()
    print(f"  Computing neighbor aggregation (A_norm @ X) ...")
    AX = np.zeros_like(X)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        AX[start:end] = A_norm[start:end] @ X
    print(f"  Done. Features: X={X.shape[1]}, AX={X.shape[1]}")
    return torch.FloatTensor(X), torch.FloatTensor(AX)


class GraphSAGENet(nn.Module):
    """
    GraphSAGE GNN with mean aggregation.
    h_v = sigma(W_self * x_v + W_neigh * MEAN(x_u : u in N(v)))
    Multi-layer with residual connections.
    """
    def __init__(self, in_channels, hidden, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # SAGE layer 1: raw features -> hidden
        self.W_self_0 = nn.Linear(in_channels, hidden)
        self.W_neigh_0 = nn.Linear(in_channels, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)

        # Additional SAGE-style layers (on hidden features, using same aggregation)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.blocks.append(nn.ModuleDict({
                'linear': nn.Linear(hidden, hidden),
                'norm': nn.BatchNorm1d(hidden),
            }))

        self.classifier = nn.Linear(hidden, 2)

    def forward(self, x_self, x_neigh):
        # SAGE layer 1
        h = self.W_self_0(x_self) + self.W_neigh_0(x_neigh)
        h = self.bn0(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Residual MLP layers
        for block in self.blocks:
            residual = h
            h = block['linear'](h)
            h = block['norm'](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + residual

        return self.classifier(h)


def train_one_model(x_self, x_neigh, data, device, config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    in_ch = x_self.shape[1]
    model = GraphSAGENet(in_ch, config['hidden'], config['layers'], config['dropout']).to(device)

    labeled = data.labeled_nodes
    train_idx = labeled[data.train_mask]
    val_idx = labeled[data.val_mask]
    y_train = data.y[data.train_mask].long()
    y_val = data.y[data.val_mask].long()

    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    pw = n_neg / max(n_pos, 1.0)
    print(f"  Class balance: neg={n_neg}, pos={n_pos}, pos_weight={pw:.2f}")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pw], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.5, min_lr=1e-6)

    best_auc, best_state, patience = 0.0, None, 0
    mb = config.get('mb', 4096)

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        perm = torch.randperm(len(train_idx))
        total_loss, nb = 0.0, 0
        for s in range(0, len(train_idx), mb):
            e = min(s + mb, len(train_idx))
            idx = perm[s:e]
            nids = train_idx[idx]
            xs = x_self[nids].to(device)
            xn = x_neigh[nids].to(device)
            y = y_train[idx].to(device)
            out = model(xs, xn)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        # Validation
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for s in range(0, len(val_idx), mb):
                e = min(s + mb, len(val_idx))
                nids = val_idx[s:e]
                xs = x_self[nids].to(device)
                xn = x_neigh[nids].to(device)
                out = model(xs, xn)
                all_p.append(torch.softmax(out, 1)[:, 1].cpu())
                all_l.append(y_val[s:e])
        vp = torch.cat(all_p).numpy()
        vl = torch.cat(all_l).numpy()
        try:
            auc = roc_auc_score(vl, vp)
        except:
            auc = 0.5

        scheduler.step(auc)
        if auc > best_auc:
            best_auc, best_state, patience = auc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [SAGE {seed}] Ep {epoch:3d} Loss={total_loss/max(nb,1):.4f} AUC={auc:.4f} Best={best_auc:.4f} LR={optimizer.param_groups[0]['lr']:.2e}")
        if patience >= config.get('es', 60):
            print(f"  [SAGE {seed}] Early stop ep {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  [SAGE {seed}] Best AUC = {best_auc:.4f}")
    return model, best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0,1,2,3,4,5,6])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    out_dir = args.model_dir or args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Dataset B ...")
    ds = load_dataset("B", args.data_dir)
    data = ds[0]
    print(f"  Nodes={data.num_nodes} Edges={data.num_edges} Features={data.x.shape[1]} Classes={ds.num_classes}")

    # Precompute SAGE mean-aggregator on CPU
    x_self, x_neigh = precompute_neighbor_agg(data)

    config = {'hidden': args.hidden, 'layers': args.layers, 'lr': args.lr,
              'wd': args.wd, 'dropout': args.dropout, 'epochs': args.epochs, 'es': 60, 'mb': 4096}

    print(f"\n{'='*60}\n  Training GraphSAGE {len(args.seeds)}-seed ensemble\n{'='*60}")

    all_probs = []
    mb = config['mb']
    for seed in args.seeds:
        print(f"\n--- SAGE Seed {seed} ---")
        model, auc = train_one_model(x_self, x_neigh, data, device, config, seed)
        logits_list = []
        with torch.no_grad():
            for s in range(0, data.num_nodes, mb):
                e = min(s + mb, data.num_nodes)
                xs = x_self[s:e].to(device)
                xn = x_neigh[s:e].to(device)
                logits_list.append(model(xs, xn).cpu())
        probs = torch.softmax(torch.cat(logits_list, 0), dim=1)
        all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    val_idx = data.labeled_nodes[data.val_mask]
    y_val = data.y[data.val_mask].long().numpy()
    try:
        ens_auc = roc_auc_score(y_val, avg_probs[val_idx, 1].numpy())
    except:
        ens_auc = 0.5

    print(f"\n{'='*60}\n  Ensemble Val AUC-ROC = {ens_auc:.4f}\n{'='*60}")

    final_logits = torch.log(avg_probs + 1e-8)
    path = os.path.join(out_dir, f"{args.kerberos}_model_B.pt")
    torch.save({"logits": final_logits}, path)
    print(f"\nModel saved to {path}\nDone!")


if __name__ == "__main__":
    main()
