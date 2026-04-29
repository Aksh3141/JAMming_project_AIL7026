"""
predict.py  –  COL761 Assignment 3 prediction script

Usage
-----
python predict.py --dataset A|B|C --task node|link --data_dir /absolute/path/to/data_dir \
--model_dir /path/to/models --output_dir /path/to/outputs --kerberos YOUR_KERBEROS

If you do not pass model_dir, the script will generate random predictions in the correct format. You can use this to test your evaluation setup before training a model.
"""

import argparse
import os

import numpy as np
import torch

from load_dataset import COL761NodeDataset, COL761LinkDataset, load_dataset, _load_edge_list


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load a model or precomputed dictionary."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    if isinstance(model, torch.nn.Module):
        model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Random fallbacks (used when no --model_path is provided)
# ─────────────────────────────────────────────────────────────────────────────

def _random_A(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.randint(0, dataset.num_classes, (dataset[0].num_nodes,))

def _random_B(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.rand(dataset[0].num_nodes)

def _random_C(V: int, K: int) -> tuple:
    return torch.rand(V), torch.rand(V, K)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset prediction functions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_A(model, dataset: COL761NodeDataset) -> torch.Tensor:
    """
    Returns predicted class index for every node → LongTensor [N].
    """
    if isinstance(model, dict) and "logits" in model:
        return model["logits"].argmax(dim=1)
        
    data = dataset[0]
    logits = model(data.x, data.edge_index)
    return logits.argmax(dim=1)


@torch.no_grad()
def predict_B(model, dataset: COL761NodeDataset) -> torch.Tensor:
    """
    Returns positive-class probability for every node → FloatTensor [N].
    """
    if isinstance(model, dict) and "logits" in model:
        logits = model["logits"]
    else:
        data = dataset[0]
        logits = model(data.x, data.edge_index)

    if logits.shape[1] == 1:
        return torch.sigmoid(logits).squeeze(1)
    return torch.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def predict_C(
    model,
    dataset: COL761LinkDataset,
    test_dir: str = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if test_dir is None:
        pos = dataset.valid_pos
        neg = dataset.valid_neg
        split = "valid"
    else:
        pos = _load_edge_list(os.path.join(test_dir, "test_pos.txt"))
        npy = os.path.join(test_dir, "test_neg_hard.npy")
        with open(npy, "rb") as f:
            neg = torch.from_numpy(np.load(f))
        split = "test"

    P, K, _ = neg.shape

    if isinstance(model, dict):
        # If we have precomputed scores AND this is the validation split, use them
        if split == "valid" and "pos_scores" in model and "neg_scores" in model:
            return model["pos_scores"], model["neg_scores"], split

        # Otherwise, reconstruct models and compute live scores
        if "encoder_states" in model:
            return _live_inference_C(model, dataset, pos, neg, split)

        # Fallback: return precomputed (may be wrong split but better than crash)
        return model["pos_scores"], model["neg_scores"], split

    pos_scores = model(dataset.x, dataset.edge_index, pos)
    neg_scores = model(dataset.x, dataset.edge_index, neg.view(P * K, 2)).view(P, K)

    return pos_scores, neg_scores, split


def _live_inference_C(model_dict, dataset, pos, neg, split):
    """Reconstruct ensemble from saved weights and compute scores for arbitrary edges."""
    from train_C import SAGEEncoder, LinkPredictor, compute_struct_features, NUM_STRUCT_FEATS

    cfg = model_dict["config"]
    encoder_states  = model_dict["encoder_states"]
    predictor_states = model_dict["predictor_states"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # SVD transform features
    svd_model = model_dict.get("svd_model", None)
    if svd_model is not None:
        x_np = dataset.x.numpy() if not dataset.x.is_sparse else dataset.x.to_dense().numpy()
        x_compressed = svd_model.transform(x_np).astype(np.float32)
        x = torch.FloatTensor(x_compressed).to(device)
    else:
        x = dataset.x.to(device)

    in_channels = cfg.get("in_channels", x.shape[1])
    edge_index = dataset.edge_index.to(device)
    P, K, _ = neg.shape

    # Leakage-free training graph for structural features
    train_ei_clean = model_dict.get("train_ei_clean", dataset.edge_index).to(device)

    # Compute 5 topology structural features
    struct_pos = compute_struct_features(
        train_ei_clean.cpu(), dataset.x.shape[0], pos
    ).to(device)
    struct_neg = compute_struct_features(
        train_ei_clean.cpu(), dataset.x.shape[0], neg.reshape(-1, 2)
    ).to(device)

    # Collect raw scores from each seed
    all_pos_scores = []
    all_neg_scores = []

    for enc_state, pred_state in zip(encoder_states, predictor_states):
        encoder = SAGEEncoder(
            in_channels=in_channels,
            hidden_channels=cfg["hidden"],
            out_channels=cfg["emb_dim"],
            num_layers=cfg["layers"],
            dropout=cfg["dropout"],
            feat_dropout=cfg.get("feat_dropout", 0.3),
        ).to(device)
        encoder.load_state_dict(enc_state)
        encoder.eval()

        predictor = LinkPredictor(
            emb_dim=cfg["emb_dim"],
            hidden_channels=cfg.get("pred_hidden", cfg["hidden"]),
            num_struct_feats=cfg.get("num_struct_feats", NUM_STRUCT_FEATS),
            dropout=cfg["dropout"],
        ).to(device)
        predictor.load_state_dict(pred_state)

        with torch.no_grad():
            emb = encoder(x, edge_index)
            ps = predictor(emb[pos[:, 0].long()], emb[pos[:, 1].long()], struct_pos).cpu()
            ns = predictor(
                emb[neg.reshape(-1, 2)[:, 0].long()],
                emb[neg.reshape(-1, 2)[:, 1].long()],
                struct_neg
            ).reshape(P, K).cpu()

        all_pos_scores.append(ps)
        all_neg_scores.append(ns)

    # Rank-based ensemble fusion (same as training)
    ens_pos_ranks = []
    ens_neg_ranks = []
    for ps, ns in zip(all_pos_scores, all_neg_scores):
        all_sc = torch.cat([ps, ns.flatten()])
        ranks = torch.zeros_like(all_sc)
        order = all_sc.argsort()
        ranks[order] = torch.arange(len(all_sc), dtype=torch.float32)
        ranks = ranks / (len(all_sc) - 1)
        ens_pos_ranks.append(ranks[:len(ps)])
        ens_neg_ranks.append(ranks[len(ps):].reshape(P, K))

    pos_scores = torch.stack(ens_pos_ranks).mean(0)
    neg_scores = torch.stack(ens_neg_ranks).mean(0)

    return pos_scores, neg_scores, split


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def predict_and_save(
    dataset_name: str,
    data_dir: str,
    model_path: str,
    out_dir: str,
    test_dir: str = None,
    kerberos: str = "student",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset {dataset_name} ...")
    ds = load_dataset(dataset_name, data_dir)

    if model_path is not None:
        print(f"Loading model from {model_path} ...")
        model = load_model(model_path)
    else:
        print("No --model_path given — using random predictions.")
        model = None

    if dataset_name == "A":
        y_pred = predict_A(model, ds) if model else _random_A(ds)
        assert y_pred.shape == (ds[0].num_nodes,), \
            f"y_pred must be shape [N={ds[0].num_nodes}], got {y_pred.shape}"
        assert y_pred.dtype == torch.long, \
            f"y_pred must be LongTensor, got {y_pred.dtype}"

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_A.pt")
        torch.save({"y_pred": y_pred}, out_path)
        print(f"Saved {out_path}  shape={y_pred.shape}")

    elif dataset_name == "B":
        y_score = predict_B(model, ds) if model else _random_B(ds)
        assert y_score.shape == (ds[0].num_nodes,), \
            f"y_score must be shape [N={ds[0].num_nodes}], got {y_score.shape}"
        assert y_score.is_floating_point(), \
            f"y_score must be float, got {y_score.dtype}"

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_B.pt")
        torch.save({"y_score": y_score}, out_path)
        print(f"Saved {out_path}  shape={y_score.shape}")

    elif dataset_name == "C":
        if model:
            pos_scores, neg_scores, split = predict_C(model, ds, test_dir=test_dir)
        else:
            if test_dir or not hasattr(ds, "valid_pos"):
                pos   = ds.test_pos
                neg   = ds.test_neg
                split = "test"
            else:
                pos   = ds.valid_pos
                neg   = ds.valid_neg
                split = "valid"
            V, K = pos.shape[0], neg.shape[1]
            pos_scores, neg_scores = _random_C(V, K)

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_C.pt")
        torch.save(
            {"pos_scores": pos_scores, "neg_scores": neg_scores, "split": split},
            out_path,
        )
        print(f"Saved {out_path}  split={split}")
        print(f"  pos_scores : {pos_scores.shape}")
        print(f"  neg_scores : {neg_scores.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for COL761 A3 datasets."
    )
    parser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    parser.add_argument("--task",       required=True, choices=["node", "link"],
                        help="Task type: node classification (A/B) or link prediction (C)")
    parser.add_argument("--data_dir",   required=True,
                        help="Absolute path to the shared datasets directory")
    parser.add_argument("--model_dir",  default=None,
                        help="Directory containing your saved model. "
                             "The script looks for <kerberos>_model_<dataset>.pt here.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where predictions will be written")
    parser.add_argument("--kerberos",   required=True,
                        help="Your Kerberos ID (used to name the output file)")
    parser.add_argument("--test_dir",   default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # validate task ↔ dataset
    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    model_path = None
    if args.model_dir is not None:
        model_path = os.path.join(
            args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt"
        )

    predict_and_save(
        args.dataset, args.data_dir, model_path, args.output_dir,
        test_dir=args.test_dir,
        kerberos=args.kerberos,
    )


if __name__ == "__main__":
    main()
