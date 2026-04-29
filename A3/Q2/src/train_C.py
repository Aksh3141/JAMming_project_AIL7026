"""
train_C.py – COL761 A3: Dataset C (Link Prediction)
Proven config: 5 topology features + margin loss + hard negatives + rank ensemble
"""
import argparse, os, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge, to_scipy_sparse_matrix
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from load_dataset import load_dataset

NUM_STRUCT_FEATS = 5

def compress_features(x_np, n_components=256, seed=0):
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    x_svd = svd.fit_transform(x_np).astype(np.float32)
    print(f"  SVD {x_np.shape[1]}→{n_components}  var: {svd.explained_variance_ratio_.sum():.3f}")
    return x_svd, svd

def build_train_edge_index(edge_index, val_pos, num_nodes):
    val_set = set()
    for u, v in val_pos.tolist():
        val_set.add((int(u), int(v))); val_set.add((int(v), int(u)))
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    keep = [i for i, (u, v) in enumerate(zip(src, dst)) if (u, v) not in val_set]
    removed = len(src) - len(keep)
    if removed > 0:
        print(f"  Removed {removed} val edges")
    return edge_index[:, torch.tensor(keep, dtype=torch.long)]

def build_hard_neg_pool(edge_index, num_nodes, pool_size=30000):
    A = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    A = A.maximum(A.T)
    A2 = A @ A
    edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    rows, cols = A2.nonzero()
    cands = [(u,v) for u,v in zip(rows,cols) if u<v and (u,v) not in edge_set and (v,u) not in edge_set]
    np.random.shuffle(cands)
    cands = cands[:pool_size]
    if not cands: return None
    print(f"  Hard negative pool: {len(cands)} 2-hop non-edges")
    return torch.tensor(cands, dtype=torch.long)

def compute_struct_features(edge_index, num_nodes, candidate_edges):
    """5 classic topology features: CN, AA, RA, Jaccard, PA"""
    nn_nodes = max(num_nodes, int(edge_index.max().item() + 1))
    A = to_scipy_sparse_matrix(edge_index, num_nodes=nn_nodes).tocsr()
    A = A.maximum(A.T)
    A2 = A @ A
    src = candidate_edges[:, 0].cpu().numpy().copy()
    dst = candidate_edges[:, 1].cpu().numpy().copy()
    deg = np.array(A.sum(axis=1)).flatten().astype(np.float64)
    cn = np.array(A2[src, dst]).flatten().astype(np.float64)
    inv_log_deg = np.zeros_like(deg)
    m = deg > 1; inv_log_deg[m] = 1.0 / np.log(deg[m])
    aa = np.array((A @ sp.diags(inv_log_deg) @ A.T)[src, dst]).flatten()
    inv_deg = np.zeros_like(deg)
    m2 = deg > 0; inv_deg[m2] = 1.0 / deg[m2]
    ra = np.array((A @ sp.diags(inv_deg) @ A.T)[src, dst]).flatten()
    ds, dd = deg[src], deg[dst]
    union = ds + dd - cn
    jaccard = np.where(union > 0, cn / (union + 1e-8), 0.0)
    pa = np.log1p(ds * dd)
    feats = np.stack([cn, aa, ra, jaccard, pa], axis=1).astype(np.float32)
    return torch.FloatTensor(feats)

class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, feat_dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.feat_dropout = feat_dropout
        self.num_layers = num_layers
        dims = [in_channels] + [hidden_channels]*(num_layers-1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.norms.append(nn.LayerNorm(dims[i+1]))
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.feat_dropout, training=self.training)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index); x = norm(x)
            if i < self.num_layers - 1:
                x = F.elu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        return F.normalize(x, p=2, dim=1)

class LinkPredictor(nn.Module):
    def __init__(self, emb_dim, hidden_channels=512, num_struct_feats=5, dropout=0.5):
        super().__init__()
        in_dim = emb_dim * 4 + num_struct_feats
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_channels), nn.LayerNorm(hidden_channels),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels//2), nn.LayerNorm(hidden_channels//2),
            nn.ELU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden_channels//2, 1),
        )
    def forward(self, emb_u, emb_v, sf=None):
        x = torch.cat([emb_u, emb_v, emb_u*emb_v, (emb_u-emb_v).abs()], dim=1)
        if sf is not None: x = torch.cat([x, sf], dim=1)
        return self.net(x).squeeze(-1)

def hits_at_k(pos_scores, neg_scores, k=50):
    return ((neg_scores > pos_scores.unsqueeze(1)).sum(dim=1) < k).float().mean().item()

def train_one_seed(dataset, device, config, seed, train_ei_clean, hard_neg_pool):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    x = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    train_ei = train_ei_clean.to(device)
    train_pos, train_neg = dataset.train_pos, dataset.train_neg
    valid_pos, valid_neg = dataset.valid_pos, dataset.valid_neg

    encoder = SAGEEncoder(x.shape[1], config['hidden'], config['emb_dim'],
                          config['layers'], config['dropout'], config['feat_dropout']).to(device)
    predictor = LinkPredictor(config['emb_dim'], config['pred_hidden'],
                              NUM_STRUCT_FEATS, config['dropout']).to(device)

    print("  Computing structural features ...")
    sf_tp = compute_struct_features(train_ei.cpu(), dataset.x.shape[0], train_pos).to(device)
    sf_tn = compute_struct_features(train_ei.cpu(), dataset.x.shape[0], train_neg).to(device)
    sf_vp = compute_struct_features(train_ei.cpu(), dataset.x.shape[0], valid_pos).to(device)
    V, K, _ = valid_neg.shape
    sf_vn = compute_struct_features(
        train_ei.cpu(), dataset.x.shape[0], valid_neg.reshape(-1,2)
    ).to(device).reshape(V, K, NUM_STRUCT_FEATS)
    if hard_neg_pool is not None:
        sf_hard = compute_struct_features(train_ei.cpu(), dataset.x.shape[0], hard_neg_pool).to(device)

    all_params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config['lr'], weight_decay=config['wd'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['T0'], T_mult=2, eta_min=1e-6)
    margin_loss = nn.MarginRankingLoss(margin=config['margin'])

    best_hits, best_states, patience_ctr = 0.0, None, 0
    n_train = train_pos.shape[0]
    bs, nnp = config['batch_size'], config['num_neg_per_pos']
    hr = config['hard_ratio']

    for epoch in range(1, config['epochs']+1):
        encoder.train(); predictor.train()
        perm = torch.randperm(n_train)
        total_loss, nb = 0.0, 0
        for start in range(0, n_train, bs):
            end = min(start+bs, n_train)
            idx = perm[start:end]; B = len(idx)
            ei_drop, _ = dropout_edge(train_ei, p=config['drop_edge_p'],
                                       force_undirected=True, training=True)
            emb = encoder(x, ei_drop)
            pos_src = train_pos[idx, 0].long().to(device)
            pos_dst = train_pos[idx, 1].long().to(device)

            n_hard = int(B*nnp*hr) if hard_neg_pool is not None else 0
            n_rand = B*nnp - n_hard

            ri = torch.randint(0, train_neg.shape[0], (n_rand,))
            ns_r = train_neg[ri, 0].long().to(device)
            nd_r = train_neg[ri, 1].long().to(device)
            sf_r = sf_tn[ri]

            if n_hard > 0:
                hi = torch.randint(0, hard_neg_pool.shape[0], (n_hard,))
                ns_h = hard_neg_pool[hi, 0].long().to(device)
                nd_h = hard_neg_pool[hi, 1].long().to(device)
                sf_h = sf_hard[hi]
                neg_src = torch.cat([ns_r, ns_h])
                neg_dst = torch.cat([nd_r, nd_h])
                sf_neg = torch.cat([sf_r, sf_h])
            else:
                neg_src, neg_dst, sf_neg = ns_r, nd_r, sf_r

            pos_sc = predictor(emb[pos_src], emb[pos_dst], sf_tp[idx])
            neg_sc = predictor(emb[neg_src], emb[neg_dst], sf_neg)
            loss = margin_loss(pos_sc.repeat_interleave(nnp), neg_sc,
                               torch.ones(B*nnp, device=device))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            total_loss += loss.item(); nb += 1

        scheduler.step()
        avg_loss = total_loss / max(nb, 1)
        encoder.eval(); predictor.eval()
        with torch.no_grad():
            emb = encoder(x, edge_index)
            vp_sc = predictor(emb[valid_pos[:,0].long().to(device)],
                              emb[valid_pos[:,1].long().to(device)], sf_vp)
            vn_list = []
            for i in range(0, K, 200):
                j = min(i+200, K)
                cs = valid_neg[:,i:j,:].reshape(-1,2)
                sf = sf_vn[:,i:j,:].reshape(-1, NUM_STRUCT_FEATS)
                vn_list.append(predictor(emb[cs[:,0].long().to(device)],
                                         emb[cs[:,1].long().to(device)], sf).reshape(V, j-i))
            vn_sc = torch.cat(vn_list, dim=1)
            val_hits = hits_at_k(vp_sc, vn_sc, k=50)

        if val_hits > best_hits:
            best_hits = val_hits
            best_states = {'encoder': copy.deepcopy(encoder.state_dict()),
                           'predictor': copy.deepcopy(predictor.state_dict())}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Seed {seed}] Ep {epoch:4d}  Loss={avg_loss:.4f}  "
                  f"Hits@50={val_hits:.4f}  Best={best_hits:.4f}  LR={optimizer.param_groups[0]['lr']:.2e}")
        if patience_ctr >= config['early_stop']:
            print(f"  [Seed {seed}] Early stop at epoch {epoch}"); break

    encoder.load_state_dict(best_states['encoder'])
    predictor.load_state_dict(best_states['predictor'])
    encoder.eval(); predictor.eval()
    print(f"  [Seed {seed}] Best Hits@50 = {best_hits:.4f}")
    return encoder, predictor, best_hits

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_dir", required=True)
    pa.add_argument("--kerberos", required=True)
    pa.add_argument("--output_dir", default=None)
    pa.add_argument("--model_dir", default=None)
    pa.add_argument("--epochs", type=int, default=600)
    pa.add_argument("--hidden", type=int, default=512)
    pa.add_argument("--pred_hidden", type=int, default=512)
    pa.add_argument("--emb_dim", type=int, default=256)
    pa.add_argument("--svd_dim", type=int, default=256)
    pa.add_argument("--layers", type=int, default=2)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--wd", type=float, default=5e-3)
    pa.add_argument("--dropout", type=float, default=0.5)
    pa.add_argument("--feat_dropout", type=float, default=0.5)
    pa.add_argument("--drop_edge_p", type=float, default=0.6)
    pa.add_argument("--margin", type=float, default=1.0)
    pa.add_argument("--hard_ratio", type=float, default=0.3)
    pa.add_argument("--batch_size", type=int, default=2048)
    pa.add_argument("--num_neg_per_pos", type=int, default=8)
    pa.add_argument("--T0", type=int, default=100)
    pa.add_argument("--seeds", type=int, nargs='+', default=[0,1,2,3,4])
    pa.add_argument("--device", default=None)
    args = pa.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    out_dir = args.model_dir or args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Dataset C ...")
    ds = load_dataset("C", args.data_dir)
    print(f"  Node features : {ds.x.shape}")
    print(f"  Edge index    : {ds.edge_index.shape}")
    raw_np = ds.x.numpy() if not ds.x.is_sparse else ds.x.to_dense().numpy()

    print(f"\nCompressing features {ds.x.shape[1]}→{args.svd_dim} via SVD ...")
    x_comp, svd_model = compress_features(raw_np, n_components=args.svd_dim)
    ds.x = torch.FloatTensor(x_comp)

    print("\nBuilding leakage-free training edge index ...")
    train_ei = build_train_edge_index(ds.edge_index, ds.valid_pos, ds.x.shape[0])
    print(f"  Full: {ds.edge_index.shape[1]}  →  Clean: {train_ei.shape[1]}")

    print("\nBuilding hard negative pool ...")
    hard_pool = build_hard_neg_pool(train_ei, ds.x.shape[0], pool_size=30000)

    config = {'hidden': args.hidden, 'pred_hidden': args.pred_hidden,
              'emb_dim': args.emb_dim, 'layers': args.layers, 'lr': args.lr,
              'wd': args.wd, 'dropout': args.dropout, 'feat_dropout': args.feat_dropout,
              'drop_edge_p': args.drop_edge_p, 'margin': args.margin,
              'hard_ratio': args.hard_ratio, 'epochs': args.epochs, 'early_stop': 100,
              'batch_size': args.batch_size, 'num_neg_per_pos': args.num_neg_per_pos, 'T0': args.T0}

    print(f"\n{'='*60}")
    print(f"  Training {len(args.seeds)}-seed ensemble")
    print(f"{'='*60}")

    all_enc, all_pred, all_hits = [], [], []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        e, p, h = train_one_seed(ds, device, config, seed, train_ei, hard_pool)
        all_enc.append(e); all_pred.append(p); all_hits.append(h)

    # Rank-based ensemble
    print("\n  Computing RANK-BASED ensemble ...")
    x_d, ei_d = ds.x.to(device), ds.edge_index.to(device)
    V, K, _ = ds.valid_neg.shape
    sf_vp = compute_struct_features(train_ei.cpu(), ds.x.shape[0], ds.valid_pos).to(device)
    sf_vn = compute_struct_features(train_ei.cpu(), ds.x.shape[0], ds.valid_neg.reshape(-1,2)).to(device)
    print(f"  Per-seed: {[f'{h:.4f}' for h in all_hits]}")

    all_ps, all_ns = [], []
    for enc, pred in zip(all_enc, all_pred):
        enc.eval(); pred.eval()
        with torch.no_grad():
            emb = enc(x_d, ei_d)
            ps = pred(emb[ds.valid_pos[:,0].long().to(device)],
                      emb[ds.valid_pos[:,1].long().to(device)], sf_vp).cpu()
            ns = pred(emb[ds.valid_neg.reshape(-1,2)[:,0].long().to(device)],
                      emb[ds.valid_neg.reshape(-1,2)[:,1].long().to(device)],
                      sf_vn).reshape(V,K).cpu()
        all_ps.append(ps); all_ns.append(ns)

    epr, enr = [], []
    for ps, ns in zip(all_ps, all_ns):
        sc = torch.cat([ps, ns.flatten()])
        r = torch.zeros_like(sc)
        o = sc.argsort()
        r[o] = torch.arange(len(sc), dtype=torch.float32)
        r /= (len(sc)-1)
        epr.append(r[:len(ps)]); enr.append(r[len(ps):].reshape(V,K))

    avg_p = torch.stack(epr).mean(0)
    avg_n = torch.stack(enr).mean(0)
    fh = hits_at_k(avg_p, avg_n, k=50)
    print(f"\n{'='*60}\n  Ensemble Hits@50 = {fh:.4f}  ({fh*100:.2f}%)\n{'='*60}")

    model_dict = {
        "pos_scores": avg_p, "neg_scores": avg_n,
        "encoder_states": [e.cpu().state_dict() for e in all_enc],
        "predictor_states": [p.cpu().state_dict() for p in all_pred],
        "weights": [1.0/len(all_enc)]*len(all_enc),
        "svd_model": svd_model, "svd_dim": args.svd_dim,
        "train_ei_clean": train_ei.cpu(),
        "config": {"encoder_type":"sage", "in_channels":args.svd_dim,
                   "hidden":config['hidden'], "pred_hidden":config['pred_hidden'],
                   "emb_dim":config['emb_dim'], "layers":config['layers'],
                   "dropout":config['dropout'], "feat_dropout":config['feat_dropout'],
                   "num_struct_feats":NUM_STRUCT_FEATS, "num_nodes":ds.x.shape[0]},
    }
    path = os.path.join(out_dir, f"{args.kerberos}_model_C.pt")
    torch.save(model_dict, path)
    print(f"\nModel saved → {path}\nDone!")

if __name__ == "__main__":
    main()
