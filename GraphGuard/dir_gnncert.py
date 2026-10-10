#!/usr/bin/env python3
"""
Train + certify GNNCert with or without direction awareness. Self-contained: does not import or modify
util.py / division.py / main_graphguard.py / test.py, so the original pipeline keeps reproducing exactly.

  --direction undirected : original GIN (models/graphcnn.py) on the symmetrised graph, edge hash on "min;max"   (= baseline)
  --direction directed   : Dir-GIN (dir_model.py) on the directed graph,               edge hash on ordered "src;dst"
  --feature_set ldp5|dldp: undirected LDP (5 dims, identical to MALNET_LDP) | directed LDP (10 dims)

Training protocol copies main_graphguard.py: batch 32 random views, 50 iters/epoch, Adam lr 1e-3, 5 layers, hidden 64,
2-layer MLPs, final dropout .5, sum pooling, StratifiedKFold(seed 42), masked features = -99, FINAL-epoch model is used.
Certification copies test.py (Mp = (N_top - N_second + I(top<second))/2 ; certified at size s  iff  correct and Mp > s).

Example (from GraphGuard/):
  python dir_gnncert.py --dataset MALNET_DIR --direction directed --feature_set dldp --division-method structure \
         --num_group 30 --epochs 1000
"""
import argparse, copy, csv, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dir_data import load_directed_dataset, prepare_groups, make_views, build_batch
from dir_model import DiGraphCNN


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MALNET_DIR")
    p.add_argument("--root", default="dataset")
    p.add_argument("--direction", choices=["undirected", "directed"], required=True)
    p.add_argument("--feature_set", choices=["ldp5", "dldp"], default="ldp5")
    p.add_argument("--division-method", dest="division", choices=["feature", "structure", "all"], required=True)
    p.add_argument("--num_group", type=int, default=30)
    p.add_argument("--hash-method", dest="hash_method", choices=["md5", "sha256", "sha512"], default="md5")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--iters_per_epoch", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold_n", type=int, default=3)
    p.add_argument("--fold_idx", type=int, default=0)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_mlp_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--final_dropout", type=float, default=0.5)
    p.add_argument("--graph_pooling_type", default="sum", choices=["sum", "average"])
    p.add_argument("--mask_value", type=float, default=-99.0, help="value written into masked node features (original: -99)")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_batch", type=int, default=512)
    p.add_argument("--checkpoint", choices=["final", "best_val"], default="final",
                   help="final = like the original code; best_val = hold out --val_frac of the TRAIN graphs and keep the best epoch")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--no_log_test", action="store_true", help="skip the periodic per-view test accuracy log")
    p.add_argument("--mode", choices=["train", "certify", "both"], default="both")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--results", default="dir_results")
    p.add_argument("--tag", default=None, help="sub-folder name, default <direction>_<feature_set>")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


@torch.no_grad()
def predict_views(model, graphs, views, args, batch):
    model.eval()
    out = []
    for i in range(0, len(views), batch):
        b = build_batch(graphs, views[i:i + batch], args.direction, args.mask_value)
        out.append(model(b).argmax(1).cpu())
    return torch.cat(out).numpy()


def view_accuracy(model, graphs, views, args):
    pred = predict_views(model, graphs, views, args, 64)
    lab = np.array([graphs[v[0]].label for v in views])
    return float((pred == lab).mean())


def votes_to_Mp(view_pred, labels, classes):
    """view_pred (n_graphs, n_views) argmax per subgraph. Same maths as test.py:get_Mp_pred."""
    votes = np.apply_along_axis(np.bincount, 1, view_pred, minlength=classes)
    idx = votes.argmax(-1)                                   # ties -> smallest class index
    out = votes.copy()
    _, cols = np.indices(out.shape)
    out[cols > idx[:, None]] -= 1                            # I(l < c)
    out.sort(axis=-1)
    Mp = (out[:, -1] - out[:, -2]) / 2
    return Mp, idx == labels, votes


def main():
    args = get_args()
    tag = args.tag or f"{args.direction}_{args.feature_set}"
    run = f"{args.dataset}_{args.division}_NG{args.num_group}_E{args.epochs}_S{args.seed}_F{args.fold_idx}"
    run_dir = os.path.join(args.results, tag, run)
    os.makedirs(run_dir, exist_ok=True)
    summ_path = os.path.join(run_dir, "summary.json"); model_path = os.path.join(run_dir, "model.pth")
    if os.path.exists(summ_path) and not args.force:
        print(f"[skip] {summ_path} exists (use --force)"); return

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    graphs, classes, label_dict = load_directed_dataset(args.dataset, args.feature_set, args.root)
    prepare_groups(graphs, args.num_group, args.hash_method)
    labels = np.array([g.label for g in graphs])
    train_idx, test_idx = list(StratifiedKFold(args.fold_n, shuffle=True, random_state=args.seed)
                               .split(np.zeros(len(labels)), labels))[args.fold_idx]    # == util.separate_data
    val_idx = None
    if args.checkpoint == "best_val":
        train_idx, val_idx = train_test_split(train_idx, test_size=args.val_frac, stratify=labels[train_idx],
                                              random_state=args.seed)
    in_dim = graphs[0].x.shape[1]
    print(f"[{tag}] {args.dataset} division={args.division} NG={args.num_group} | graphs={len(graphs)} "
          f"train={len(train_idx)} test={len(test_idx)} val={0 if val_idx is None else len(val_idx)} "
          f"classes={classes} feat_dim={in_dim} device={device}")

    if args.direction == "undirected":
        from models.graphcnn import GraphCNN                       # the ORIGINAL model, unchanged
        model = GraphCNN(args.num_layers, args.num_mlp_layers, in_dim, args.hidden_dim, classes, args.final_dropout,
                         False, args.graph_pooling_type, "sum", device).to(device)
    else:
        model = DiGraphCNN(args.num_layers, args.num_mlp_layers, in_dim, args.hidden_dim, classes, args.final_dropout,
                           args.graph_pooling_type, device).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    train_views = make_views(train_idx, args.division, args.num_group)
    test_views = make_views(test_idx, args.division, args.num_group)
    val_views = make_views(val_idx, args.division, args.num_group) if val_idx is not None else None
    t_train = 0.0

    if args.mode in ("train", "both"):
        crit = nn.CrossEntropyLoss(); opt = optim.Adam(model.parameters(), lr=args.lr)
        log = open(os.path.join(run_dir, "train_log.csv"), "w", newline=""); lw = csv.writer(log)
        lw.writerow(["epoch", "avg_loss", "test_view_acc", "val_view_acc"])
        best, best_state, best_ep = -1.0, None, 0
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train(); acc_loss = 0.0
            for _ in range(args.iters_per_epoch):
                bv = random.sample(train_views, args.batch_size)
                out = model(build_batch(graphs, bv, args.direction, args.mask_value))
                y = torch.LongTensor([graphs[v[0]].label for v in bv]).to(device)
                loss = crit(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
                acc_loss += loss.item()
            avg = acc_loss / args.iters_per_epoch
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                ta = "" if args.no_log_test else view_accuracy(model, graphs, test_views, args)
                va = ""
                if val_views is not None:
                    va = view_accuracy(model, graphs, val_views, args)
                    if va > best:
                        best, best_ep, best_state = va, epoch, copy.deepcopy(model.state_dict())
                lw.writerow([epoch, f"{avg:.5f}", ta, va]); log.flush()
                print(f"epoch {epoch:5d} loss {avg:.4f} test_view_acc {ta} val_view_acc {va}", flush=True)
        log.close()
        t_train = time.time() - t0
        if args.checkpoint == "best_val" and best_state is not None:
            model.load_state_dict(best_state); print(f"using best-val checkpoint from epoch {best_ep}")
        torch.save(model.state_dict(), model_path)

    if args.mode in ("certify", "both"):
        if args.mode == "certify":
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        nv = len(test_views) // len(test_idx)
        vp = predict_views(model, graphs, test_views, args, args.eval_batch).reshape(len(test_idx), nv)
        lab = labels[test_idx]
        Mp, pred, votes = votes_to_Mp(vp, lab, classes)
        n = len(Mp)
        if args.division == "all":
            R = 2 + args.num_group // 2
            M = np.zeros((R, R))
            for f in range(R):
                for s in range(R):
                    size = (f + s) * args.num_group - f * s
                    M[f, s] = np.sum(pred & (Mp > size)) / n
            np.savetxt(os.path.join(run_dir, "certified_matrix.csv"), M, delimiter=",", fmt="%.6f")
            print("certified accuracy matrix (rows = #feature perturbations, cols = #edge perturbations):")
            np.set_printoptions(precision=4, linewidth=200); print(M)
        else:
            sizes = list(range(int(Mp.max() + 1)))
            accs = [float(np.sum(pred & (Mp > s)) / n) for s in sizes]
            with open(os.path.join(run_dir, "certified_curve.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["size", "acc"]); w.writerows(zip(sizes, accs))
            print("certified accuracy @ size 0,1,2,...:", np.round(accs, 4).tolist())
        np.savez(os.path.join(run_dir, "per_graph.npz"), test_idx=test_idx, labels=lab, Mp=Mp, pred=pred, votes=votes)
        summary = dict(vars(args), tag=tag, run=run, n_test=n, clean_vote_acc=float(pred.mean()),
                       view_acc_test=float(np.mean(vp == lab[:, None])), train_seconds=t_train,
                       label_dict={str(k): v for k, v in label_dict.items()}, feat_dim=in_dim,
                       test_idx=[int(i) for i in test_idx])
        json.dump(summary, open(summ_path, "w"), indent=1)
        print(f"clean accuracy (majority vote over {nv} subgraphs): {pred.mean():.4f}  -> saved to {run_dir}")


if __name__ == "__main__":
    main()
