"""
train_C.py  –  COL761 Assignment 3: Dataset C (Link Prediction)

Architecture : GATv2 encoder + Deep MLP link predictor
Features     : 6 structural features (CN, AA, RA, Jaccard, PA, Katz)
Loss         : Multi-negative BPR Loss with margin
Boosters     : 7-seed weighted ensemble
Metric       : Hits@50
"""

import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp

from load_dataset import load_dataset

NUM_STRUCT_FEATS = 6


# ─────────────────────────────────────────────────────────────────────────────
#  6 Structural features
# ─────────────────────────────────────────────────────────────────────────────

def compute_struct_features(edge_index, num_nodes, candidate_edges):
    """
    Compute 6 structural heuristic features for candidate edges.
    candidate_edges: Tensor [E, 2]
    Returns: Tensor [E, 6] with (CN, AA, RA, Jaccard, PA, Katz)
    """
    nn_nodes = max(num_nodes, int(edge_index.max().item() + 1))
    A = to_scipy_sparse_matrix(edge_index, num_nodes=nn_nodes).tocsr()
    A2 = A @ A

    src = candidate_edges[:, 0].cpu().numpy().copy()
    dst = candidate_edges[:, 1].cpu().numpy().copy()

    # 1. Common Neighbors
    cn = np.array(A2[src, dst]).flatten().astype(np.float64)

    deg = np.array(A.sum(axis=1)).flatten().astype(np.float64)

    # 2. Adamic-Adar
    inv_log_deg = np.zeros_like(deg)
    mask = deg > 1
    inv_log_deg[mask] = 1.0 / np.log(deg[mask])
    D_aa = sp.diags(inv_log_deg)
    AA_mat = A @ D_aa @ A.T
    aa = np.array(AA_mat[src, dst]).flatten()

    # 3. Resource Allocation
    inv_deg = np.zeros_like(deg)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    D_ra = sp.diags(inv_deg)
    RA_mat = A @ D_ra @ A.T
    ra = np.array(RA_mat[src, dst]).flatten()

    # 4. Jaccard Coefficient: CN / |N(u) ∪ N(v)|
    # |N(u) ∪ N(v)| = deg(u) + deg(v) - CN
    deg_src = deg[src]
    deg_dst = deg[dst]
    union = deg_src + deg_dst - cn
    jaccard = np.where(union > 0, cn / union, 0.0)

    # 5. Preferential Attachment: deg(u) * deg(v)
    pa = deg_src * deg_dst
    # Log-scale to avoid huge values
    pa = np.log1p(pa)

    # 6. Katz Index (2-hop approx): β*A + β²*A²
    beta = 0.01
    A3 = A2 @ A
    katz_2 = beta * np.array(A[src, dst]).flatten() + \
             beta**2 * np.array(A2[src, dst]).flatten() + \
             beta**3 * np.array(A3[src, dst]).flatten()

    feats = np.stack([cn, aa, ra, jaccard, pa, katz_2], axis=1).astype(np.float32)
    return torch.FloatTensor(feats)


# ─────────────────────────────────────────────────────────────────────────────
#  GATv2 Encoder
# ─────────────────────────────────────────────────────────────────────────────

class GATv2Encoder(nn.Module):
    """GATv2 encoder for learning node embeddings."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        if num_layers == 1:
            self.convs.append(GATv2Conv(in_channels, out_channels, heads=1, concat=False, dropout=dropout))
            self.norms.append(nn.LayerNorm(out_channels))
            self.res_projs.append(
                nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
            )
        else:
            # First
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
            self.res_projs.append(
                nn.Linear(in_channels, hidden_channels * heads)
                if in_channels != hidden_channels * heads else nn.Identity()
            )
            # Middle
            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout))
                self.norms.append(nn.LayerNorm(hidden_channels * heads))
                self.res_projs.append(nn.Identity())
            # Last
            self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
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
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  MLP Link Predictor
# ─────────────────────────────────────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """
    MLP: [emb_u || emb_v || emb_u*emb_v || |emb_u-emb_v| || 6 struct feats] → score
    """
    def __init__(self, emb_dim, hidden_channels=512, num_struct_feats=6, dropout=0.3):
        super().__init__()
        in_dim = emb_dim * 4 + num_struct_feats   # concat, hadamard, abs-diff + struct
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.LayerNorm(hidden_channels // 4),
            nn.ELU(),
            nn.Linear(hidden_channels // 4, 1),
        )

    def forward(self, emb_u, emb_v, struct_feats=None):
        x = torch.cat([emb_u, emb_v, emb_u * emb_v, (emb_u - emb_v).abs()], dim=1)
        if struct_feats is not None:
            x = torch.cat([x, struct_feats], dim=1)
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-negative BPR Loss
# ─────────────────────────────────────────────────────────────────────────────

def bpr_loss_multi_neg(pos_scores, neg_scores_multi, margin=0.1):
    """
    pos_scores: [B]
    neg_scores_multi: [B, K] — K negatives per positive
    Take hardest negative for each positive.
    """
    hardest_neg, _ = neg_scores_multi.max(dim=1)  # [B]
    return -F.logsigmoid(pos_scores - hardest_neg - margin).mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Hits@K evaluation
# ─────────────────────────────────────────────────────────────────────────────

def hits_at_k(pos_scores, neg_scores, k=50):
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop (single seed)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_seed(dataset, device, config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x          = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    train_pos  = dataset.train_pos
    train_neg  = dataset.train_neg
    valid_pos  = dataset.valid_pos
    valid_neg  = dataset.valid_neg

    in_channels = x.shape[1]
    emb_dim     = config['emb_dim']
    n_neg_per_pos = config.get('n_neg', 5)

    encoder = GATv2Encoder(
        in_channels=in_channels,
        hidden_channels=config['hidden'],
        out_channels=emb_dim,
        num_layers=config['layers'],
        heads=config.get('heads', 4),
        dropout=config['dropout'],
    ).to(device)

    predictor = LinkPredictor(
        emb_dim=emb_dim,
        hidden_channels=config['hidden'],
        num_struct_feats=NUM_STRUCT_FEATS,
        dropout=config['dropout'],
    ).to(device)

    # Precompute structural features
    print("  Computing structural features for training edges ...")
    struct_pos = compute_struct_features(edge_index.cpu(), dataset.num_nodes, train_pos).to(device)
    struct_neg = compute_struct_features(edge_index.cpu(), dataset.num_nodes, train_neg).to(device)

    print("  Computing structural features for validation edges ...")
    struct_val_pos = compute_struct_features(edge_index.cpu(), dataset.num_nodes, valid_pos).to(device)
    V, K, _ = valid_neg.shape
    valid_neg_flat = valid_neg.reshape(-1, 2)
    struct_val_neg = compute_struct_features(
        edge_index.cpu(), dataset.num_nodes, valid_neg_flat
    ).to(device).reshape(V, K, NUM_STRUCT_FEATS)

    all_params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer  = torch.optim.Adam(all_params, lr=config['lr'], weight_decay=config['wd'])
    scheduler  = ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5, min_lr=1e-6)

    best_hits            = 0.0
    best_encoder_state   = None
    best_predictor_state = None
    patience_counter     = 0
    early_stop_patience  = config.get('early_stop', 60)

    n_train    = train_pos.shape[0]
    batch_size = config.get('batch_size', 4096)

    for epoch in range(1, config['epochs'] + 1):
        encoder.train()
        predictor.train()

        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches  = 0

        emb = encoder(x, edge_index)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            B = len(idx)

            pos_src = train_pos[idx, 0].long().to(device)
            pos_dst = train_pos[idx, 1].long().to(device)

            # Multi-negative: sample n_neg negatives per positive
            neg_indices = torch.randint(0, train_neg.shape[0], (B, n_neg_per_pos))
            neg_src = train_neg[neg_indices, 0].long().to(device)  # [B, K]
            neg_dst = train_neg[neg_indices, 1].long().to(device)  # [B, K]

            pos_u_emb = emb[pos_src]
            pos_v_emb = emb[pos_dst]
            pos_struct = struct_pos[idx]

            pos_scores = predictor(pos_u_emb, pos_v_emb, pos_struct)

            # Score all negatives
            neg_scores_list = []
            for k in range(n_neg_per_pos):
                ns = neg_src[:, k]
                nd = neg_dst[:, k]
                neg_struct_k = struct_neg[neg_indices[:, k]]
                s = predictor(emb[ns], emb[nd], neg_struct_k)
                neg_scores_list.append(s)
            neg_scores_multi = torch.stack(neg_scores_list, dim=1)  # [B, K]

            loss = bpr_loss_multi_neg(pos_scores, neg_scores_multi,
                                       margin=config.get('margin', 0.1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            emb = encoder(x, edge_index)

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validate
        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            emb = encoder(x, edge_index)

            vp_src = valid_pos[:, 0].long().to(device)
            vp_dst = valid_pos[:, 1].long().to(device)
            val_pos_scores = predictor(emb[vp_src], emb[vp_dst], struct_val_pos)

            val_neg_scores_list = []
            neg_batch = 100
            for i in range(0, K, neg_batch):
                j          = min(i + neg_batch, K)
                chunk      = valid_neg[:, i:j, :]
                chunk_flat = chunk.reshape(-1, 2)
                cs         = chunk_flat[:, 0].long().to(device)
                cd         = chunk_flat[:, 1].long().to(device)
                chunk_struct = struct_val_neg[:, i:j, :].reshape(-1, NUM_STRUCT_FEATS)
                scores = predictor(emb[cs], emb[cd], chunk_struct).reshape(V, j - i)
                val_neg_scores_list.append(scores)

            val_neg_scores = torch.cat(val_neg_scores_list, dim=1)
            val_hits = hits_at_k(val_pos_scores, val_neg_scores, k=50)

        scheduler.step(val_hits)

        if val_hits > best_hits:
            best_hits            = val_hits
            best_encoder_state   = copy.deepcopy(encoder.state_dict())
            best_predictor_state = copy.deepcopy(predictor.state_dict())
            patience_counter     = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [Seed {seed}] Epoch {epoch:4d}  Loss={avg_loss:.4f}  "
                  f"Hits@50={val_hits:.4f}  Best={best_hits:.4f}  LR={lr_now:.2e}")

        if patience_counter >= early_stop_patience:
            print(f"  [Seed {seed}] Early stopping at epoch {epoch}")
            break

    encoder.load_state_dict(best_encoder_state)
    predictor.load_state_dict(best_predictor_state)
    encoder.eval()
    predictor.eval()
    print(f"  [Seed {seed}] Best Hits@50 = {best_hits:.4f}")
    return encoder, predictor, best_hits


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train model for Dataset C")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--n_neg", type=int, default=5, help="Negatives per positive")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir = args.model_dir or args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Dataset C ...")
    ds = load_dataset("C", args.data_dir)
    print(f"  {ds}")
    print(f"  Node features: {ds.x.shape}")
    print(f"  Edge index: {ds.edge_index.shape}")
    print(f"  Train pos: {ds.train_pos.shape}")
    print(f"  Train neg: {ds.train_neg.shape}")
    print(f"  Valid pos: {ds.valid_pos.shape}")
    print(f"  Valid neg: {ds.valid_neg.shape}")

    config = {
        'hidden':     args.hidden,
        'emb_dim':    args.emb_dim,
        'layers':     args.layers,
        'heads':      args.heads,
        'lr':         args.lr,
        'wd':         args.wd,
        'dropout':    args.dropout,
        'epochs':     args.epochs,
        'early_stop': 60,
        'batch_size': args.batch_size,
        'margin':     args.margin,
        'n_neg':      args.n_neg,
    }

    print(f"\n{'='*60}")
    print(f"  Training {len(args.seeds)}-seed ensemble")
    print(f"{'='*60}")

    all_encoders   = []
    all_predictors = []
    all_hits       = []

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        encoder, predictor, val_hits = train_one_seed(ds, device, config, seed)
        all_encoders.append(encoder)
        all_predictors.append(predictor)
        all_hits.append(val_hits)

    # ------------------------------------------------------------------
    # Ensemble inference
    # ------------------------------------------------------------------
    print(f"\n  Computing ensemble scores from all {len(args.seeds)} seeds ...")
    x          = ds.x.to(device)
    edge_index = ds.edge_index.to(device)
    V, K, _    = ds.valid_neg.shape

    ensemble_pos_scores = []
    ensemble_neg_scores = []

    for encoder, predictor in zip(all_encoders, all_predictors):
        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            emb = encoder(x, edge_index)

            struct_pos = compute_struct_features(
                ds.edge_index.cpu(), ds.num_nodes, ds.valid_pos
            ).to(device)
            struct_neg = compute_struct_features(
                ds.edge_index.cpu(), ds.num_nodes, ds.valid_neg.reshape(-1, 2)
            ).to(device)

            pos_scores = predictor(
                emb[ds.valid_pos[:, 0].long()],
                emb[ds.valid_pos[:, 1].long()],
                struct_pos
            )
            neg_scores = predictor(
                emb[ds.valid_neg.reshape(-1, 2)[:, 0].long()],
                emb[ds.valid_neg.reshape(-1, 2)[:, 1].long()],
                struct_neg
            ).reshape(V, K)

            ensemble_pos_scores.append(pos_scores.cpu())
            ensemble_neg_scores.append(neg_scores.cpu())

    # Weighted ensemble
    hits_arr = np.array(all_hits, dtype=np.float32)
    hits_arr = np.maximum(hits_arr, 0.01)
    weights  = hits_arr / hits_arr.sum()
    print(f"  Seed weights: {[f'{w:.3f}' for w in weights]}")

    avg_pos_scores = torch.stack(
        [s * w for s, w in zip(ensemble_pos_scores, weights)]
    ).sum(dim=0)
    avg_neg_scores = torch.stack(
        [s * w for s, w in zip(ensemble_neg_scores, weights)]
    ).sum(dim=0)

    final_hits = hits_at_k(avg_pos_scores, avg_neg_scores, k=50)

    print(f"\n{'='*60}")
    print(f"  Ensemble Hits@50 = {final_hits:.4f}  ({final_hits*100:.2f}%)")
    print(f"  Per-seed best: {[f'{h:.4f}' for h in all_hits]}")
    print(f"{'='*60}")

    model_dict = {
        # Precomputed validation scores (for fast local evaluation)
        "pos_scores": avg_pos_scores,
        "neg_scores": avg_neg_scores,
        # Model weights for live inference on test data
        "encoder_states": [enc.cpu().state_dict() for enc in all_encoders],
        "predictor_states": [pred.cpu().state_dict() for pred in all_predictors],
        "weights": weights.tolist(),
        "config": {
            "in_channels": ds.x.shape[1],
            "hidden": config['hidden'],
            "emb_dim": config['emb_dim'],
            "layers": config['layers'],
            "heads": config.get('heads', 4),
            "dropout": config['dropout'],
            "num_struct_feats": NUM_STRUCT_FEATS,
        },
    }
    model_path = os.path.join(out_dir, f"{args.kerberos}_model_C.pt")
    torch.save(model_dict, model_path)
    print(f"\nModel saved to {model_path}")
    print("Done!")


if __name__ == "__main__":
    main()
