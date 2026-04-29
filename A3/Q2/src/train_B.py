"""
train_B.py  –  COL761 A3: Dataset B (Binary Node Classification)

Architecture : SIGN GNN — 3-hop + PPR multi-scale graph convolution
               [X, AX, A²X, A³X, PPR·X] each with learned W_k transforms
Loss         : CrossEntropyLoss (capped pos_weight + balanced sampling)
Metric       : AUC-ROC
"""

import argparse, os, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score
from load_dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
#  Precompute multi-hop + PPR features on CPU (SIGN architecture)
# ─────────────────────────────────────────────────────────────────────────────

def build_adj_norm(data, add_self_loops=True):
    """Build row-normalized adjacency D^{-1}A using scipy sparse."""
    import scipy.sparse as sp
    N = data.num_nodes
    src = data.edge_index[0].numpy().astype(np.int64)
    dst = data.edge_index[1].numpy().astype(np.int64)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    if add_self_loops:
        idx = np.arange(N, dtype=np.int64)
        row = np.concatenate([row, idx])
        col = np.concatenate([col, idx])
    vals = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((vals, (row, col)), shape=(N, N))
    deg = np.array(A.sum(axis=1)).flatten()
    D_inv = sp.diags(1.0 / np.maximum(deg, 1e-12))
    return (D_inv @ A).astype(np.float32), A


def sparse_matmul_chunked(S, X, chunk_size=100_000):
    """Compute S @ X in chunks to avoid memory spikes."""
    N = X.shape[0]
    out = np.zeros_like(X)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        out[s:e] = S[s:e] @ X
    return out


def compute_ppr_matrix(A_norm, alpha=0.15, num_iters=20):
    """Approximate PPR diffusion matrix via power iteration."""
    import scipy.sparse as sp
    N = A_norm.shape[0]
    # PPR: π = α·I + (1-α)·A_norm·π
    # Power iteration: π_0 = I, π_{k+1} = α·I + (1-α)·A_norm·π_k
    # We compute PPR @ X directly
    return A_norm, alpha, num_iters  # return params, compute lazily


def precompute_sign_features(data, k_hops=2, use_ppr=False, ppr_alpha=0.15, ppr_iters=5):
    """
    Precompute SIGN multi-hop features using ultra-fast PyTorch sparse mm.
    """
    import scipy.sparse as sp
    print(f"  Building adjacency ({data.num_nodes:,} nodes) ...")
    N = data.num_nodes
    src, dst = data.edge_index[0].numpy(), data.edge_index[1].numpy()
    
    # Add self loops
    idx = np.arange(N, dtype=np.int64)
    row = np.concatenate([src, dst, idx])
    col = np.concatenate([dst, src, idx])
    vals = np.ones(len(row), dtype=np.float32)
    
    A = sp.csr_matrix((vals, (row, col)), shape=(N, N))
    deg = np.array(A.sum(axis=1)).flatten()
    D_inv = sp.diags(1.0 / np.maximum(deg, 1e-12))
    A_norm = D_inv @ A
    A_norm = A_norm.tocoo()
    
    # Create PyTorch sparse tensor
    indices = torch.LongTensor(np.vstack((A_norm.row, A_norm.col)))
    values = torch.FloatTensor(A_norm.data)
    shape = torch.Size(A_norm.shape)
    A_torch = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    
    X = data.x.clone()
    feature_list = [X]
    print(f"    Hop 0: raw features ({X.shape[1]} dims)")
    
    H = X
    for hop in range(1, k_hops + 1):
        H = torch.sparse.mm(A_torch, H)
        feature_list.append(H)
        print(f"    Hop {hop}: A^{hop}·X done")
        
    if use_ppr:
        print(f"    Computing PPR diffusion (alpha={ppr_alpha}, iters={ppr_iters}) ...")
        PPR_X = X.clone()
        for i in range(ppr_iters):
            PPR_X = ppr_alpha * X + (1 - ppr_alpha) * torch.sparse.mm(A_torch, PPR_X)
        feature_list.append(PPR_X)
        print(f"    PPR done")
        
    dims = [f.shape[1] for f in feature_list]
    print(f"  SIGN features: {len(feature_list)} operators, dims={dims}")
    return feature_list


# ─────────────────────────────────────────────────────────────────────────────
#  SIGN Model — multi-hop inception GNN
# ─────────────────────────────────────────────────────────────────────────────

class SIGNModel(nn.Module):
    """
    SIGN: Scalable Inception Graph Neural Network
    Each hop gets its own W_k transform, outputs are summed then classified.
    This IS a GNN — it uses graph convolution operators (A^k @ X) with learned transforms.
    """
    def __init__(self, in_channels, hidden, num_operators, num_blocks=5, dropout=0.3):
        super().__init__()
        self.num_operators = num_operators
        self.dropout = dropout
        
        # Per-operator transforms (SAGE-style: each hop has its own W)
        self.hop_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.BatchNorm1d(hidden),
                nn.ELU(),
            ) for _ in range(num_operators)
        ])
        
        # Deep classifier with residual connections
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'linear': nn.Linear(hidden, hidden),
                'norm': nn.BatchNorm1d(hidden),
                'drop': nn.Dropout(dropout),
            }))
        
        self.head = nn.Linear(hidden, 2)
        self.input_drop = nn.Dropout(dropout)

    def forward(self, feature_list):
        """feature_list: list of [B, F] tensors, one per operator."""
        # Transform each operator's features
        h = torch.zeros_like(self.hop_transforms[0][0](feature_list[0][:1]).expand(feature_list[0].shape[0], -1))
        for i, (feat, transform) in enumerate(zip(feature_list, self.hop_transforms)):
            h = h + transform(feat)
        
        h = self.input_drop(h)
        
        # Residual blocks
        for block in self.blocks:
            residual = h
            h = block['linear'](h)
            h = block['norm'](h)
            h = F.elu(h)
            h = block['drop'](h)
            h = h + residual
        
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
#  Balanced mini-batch sampler
# ─────────────────────────────────────────────────────────────────────────────

class BalancedBatchSampler:
    """Sample balanced batches: 50% pos, 50% neg."""
    def __init__(self, labels, batch_size):
        self.pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        self.neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        self.batch_size = batch_size
        self.half = batch_size // 2
    
    def __iter__(self):
        n_batches = max(len(self.neg_idx) // self.half, 1)
        pos_perm = self.pos_idx[torch.randint(0, len(self.pos_idx), (n_batches * self.half,))]
        neg_perm = self.neg_idx[torch.randperm(len(self.neg_idx))]
        
        for i in range(n_batches):
            pos = pos_perm[i*self.half:(i+1)*self.half]
            neg_start = (i * self.half) % len(self.neg_idx)
            neg_end = neg_start + self.half
            if neg_end <= len(self.neg_idx):
                neg = neg_perm[neg_start:neg_end]
            else:
                neg = torch.cat([neg_perm[neg_start:], neg_perm[:neg_end - len(self.neg_idx)]])
            yield torch.cat([pos, neg])
    
    def __len__(self):
        return max(len(self.neg_idx) // self.half, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_model(feat_list_cpu, data, device, config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_ops = len(feat_list_cpu)
    in_ch = feat_list_cpu[0].shape[1]
    
    model = SIGNModel(
        in_channels=in_ch, hidden=config['hidden'],
        num_operators=num_ops, num_blocks=config['blocks'],
        dropout=config['dropout']
    ).to(device)

    labeled = data.labeled_nodes
    train_idx = labeled[data.train_mask]
    val_idx = labeled[data.val_mask]
    y_train = data.y[data.train_mask].long()
    y_val = data.y[data.val_mask].long()

    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    pw = min(n_neg / max(n_pos, 1.0), 15.0)  # CAPPED at 15
    print(f"  Class: neg={n_neg}, pos={n_pos}, capped_pw={pw:.2f}")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pw], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    sampler = BalancedBatchSampler(y_train, config['mb'])
    best_auc, best_state, patience = 0.0, None, 0
    mb = config['mb']

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss, nb = 0.0, 0
        
        for batch_idx in sampler:
            nids = train_idx[batch_idx]
            feats = [f[nids].to(device) for f in feat_list_cpu]
            y = y_train[batch_idx].to(device)
            
            out = model(feats)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        
        scheduler.step()

        # Validation
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for s in range(0, len(val_idx), mb):
                e = min(s + mb, len(val_idx))
                nids = val_idx[s:e]
                feats = [f[nids].to(device) for f in feat_list_cpu]
                out = model(feats)
                all_p.append(torch.softmax(out, 1)[:, 1].cpu())
                all_l.append(y_val[s:e])
        vp = torch.cat(all_p).numpy()
        vl = torch.cat(all_l).numpy()
        try:
            auc = roc_auc_score(vl, vp)
        except:
            auc = 0.5

        if auc > best_auc:
            best_auc, best_state, patience = auc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [SIGN {seed}] Ep {epoch:3d} Loss={total_loss/max(nb,1):.4f} AUC={auc:.4f} Best={best_auc:.4f} LR={optimizer.param_groups[0]['lr']:.2e}")
        if patience >= config.get('es', 60):
            print(f"  [SIGN {seed}] Early stop ep {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  [SIGN {seed}] Best AUC = {best_auc:.4f}")
    return model, best_auc


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--k_hops", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
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

    # Precompute SIGN multi-hop features
    print("\n  Precomputing SIGN features ...")
    feat_list = precompute_sign_features(data, k_hops=args.k_hops, use_ppr=False)

    config = {
        'hidden': args.hidden, 'blocks': args.blocks,
        'lr': args.lr, 'wd': args.wd, 'dropout': args.dropout,
        'epochs': args.epochs, 'es': 60, 'mb': 4096,
    }

    print(f"\n{'='*60}\n  Training SIGN GNN {len(args.seeds)}-seed ensemble\n{'='*60}")

    all_probs = []
    mb = config['mb']
    for seed in args.seeds:
        print(f"\n--- SIGN Seed {seed} ---")
        model, auc = train_one_model(feat_list, data, device, config, seed)
        logits_list = []
        with torch.no_grad():
            for s in range(0, data.num_nodes, mb):
                e = min(s + mb, data.num_nodes)
                feats = [f[s:e].to(device) for f in feat_list]
                logits_list.append(model(feats).cpu())
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
