#!/usr/bin/env python3
"""
One-time conversion: MalNet-Tiny (.edgelist, no node features)  ->  GNNCert text format
with Local Degree Profile (LDP) node features.

LDP(v) = ( d(v), min N(v), max N(v), mean N(v), std N(v) )   where N(v) = {d(u) : u adjacent to v}
(Cai & Wang, "A simple yet effective baseline for non-attribute graph classification")

Steps
  1. scan  <src>/<family>/*.edgelist  (each family sub-folder = one class)
  2. keep only graphs with  nodes < --max-nodes  (default 500)
  3. pick the SAME number of graphs from every family (random, seeded)
  4. directed -> undirected, drop self loops / duplicates, relabel node ids to 0..n-1
  5. compute LDP, write  <out>/<name>.txt  in the same layout as MUTAG.txt, with the 5 LDP
     values appended to every node line as node attributes:

        N_graphs
        n label
        tag num_neighbors nb_1 ... nb_k  ldp_1 ldp_2 ldp_3 ldp_4 ldp_5     (one line per node)

Run from  GNNCERT/GraphGuard :
    python prepare_malnet_ldp.py --total 500
"""
import argparse, csv, os, random, sys
import numpy as np


def read_header_nodes(path):
    """Fast: read '# Nodes: N, Edges: M' from the comment header. Returns N or None."""
    with open(path, "r") as f:
        for _ in range(12):
            line = f.readline()
            if not line.startswith("#"):
                break
            if "Nodes:" in line:
                try:
                    return int(line.split("Nodes:")[1].split(",")[0].strip())
                except ValueError:
                    return None
    return None


def parse_edgelist(path):
    """Return undirected, de-duplicated, relabelled edge array (E,2) and n."""
    raw = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            a = line.split()
            if len(a) < 2:
                continue
            u, v = int(a[0]), int(a[1])
            if u != v:                                  # drop self loops
                raw.append((min(u, v), max(u, v)))      # undirected
    if not raw:
        return np.zeros((0, 2), dtype=np.int64), 0
    e = np.unique(np.array(raw, dtype=np.int64), axis=0)  # remove duplicate / reverse edges
    ids, inv = np.unique(e.ravel(), return_inverse=True)  # ids are NOT contiguous in MalNet -> relabel
    return inv.reshape(-1, 2), len(ids)


def local_degree_profile(edges, n):
    """(n,5) array: degree, min/max/mean/std of neighbour degrees."""
    u = np.concatenate([edges[:, 0], edges[:, 1]])
    v = np.concatenate([edges[:, 1], edges[:, 0]])
    deg = np.bincount(u, minlength=n).astype(np.float64)
    nd = deg[v]
    mn = np.full(n, np.inf); np.minimum.at(mn, u, nd)
    mx = np.full(n, -np.inf); np.maximum.at(mx, u, nd)
    s1 = np.bincount(u, weights=nd, minlength=n)
    s2 = np.bincount(u, weights=nd * nd, minlength=n)
    safe = np.maximum(deg, 1)
    mean = s1 / safe
    std = np.sqrt(np.maximum(s2 / safe - mean ** 2, 0.0))
    iso = deg == 0
    mn[iso] = mx[iso] = mean[iso] = std[iso] = 0.0
    return np.stack([deg, mn, mx, mean, std], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/malnet-graphs-tiny", help="folder with one sub-folder per family")
    ap.add_argument("--out", default="dataset", help="dataset root (output goes to <out>/<name>/<name>.txt)")
    ap.add_argument("--name", default="MALNET_LDP")
    ap.add_argument("--total", type=int, default=500, help="total graphs (split equally over families), e.g. 250 or 500")
    ap.add_argument("--max-nodes", type=int, default=650, help="keep graphs with nodes < this value")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    a = ap.parse_args()

    out_dir = os.path.join(a.out, a.name)
    out_txt = os.path.join(out_dir, a.name + ".txt")
    if os.path.exists(out_txt) and not a.force:
        print(f"[skip] {out_txt} already exists (use --force to rebuild)")
        return

    families = sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)))
    if not families:
        sys.exit(f"No family sub-folders found in {a.src}")
    per_class = a.total // len(families)
    print(f"families (label order): {dict(enumerate(families))}")
    print(f"target: {per_class} graphs per family  ({per_class * len(families)} total), nodes < {a.max_nodes}")

    rng = random.Random(a.seed)
    chosen = []  # (label, family, path, edges, n)
    for label, fam in enumerate(families):
        fdir = os.path.join(a.src, fam)
        files = sorted(f for f in os.listdir(fdir) if f.endswith(".edgelist"))
        cand = []
        for fn in files:                                   # cheap header scan
            p = os.path.join(fdir, fn)
            n_hdr = read_header_nodes(p)
            if n_hdr is None or n_hdr < a.max_nodes:
                cand.append(p)
        rng.shuffle(cand)
        got = []
        for p in cand:                                     # verify after parsing, stop at quota
            edges, n = parse_edgelist(p)
            if n == 0 or n >= a.max_nodes:
                continue
            got.append((label, fam, p, edges, n))
            if len(got) == per_class:
                break
        print(f"  {fam:12s} files={len(files):5d}  eligible(<{a.max_nodes})={len(cand):5d}  selected={len(got)}")
        if len(got) < per_class:
            print(f"  [warn] {fam}: only {len(got)} eligible graphs, wanted {per_class} "
                  f"-> classes will be unbalanced. Lower --total or raise --max-nodes.")
        chosen.extend(got)

    if not chosen:
        sys.exit('No graphs selected - check --src and --max-nodes.')
    rng.shuffle(chosen)  # mix classes (StratifiedKFold stratifies anyway)
    os.makedirs(out_dir, exist_ok=True)
    allfeat = []
    with open(out_txt, "w") as f, open(os.path.join(out_dir, a.name + "_files.csv"), "w", newline="") as m:
        w = csv.writer(m); w.writerow(["index", "label", "family", "nodes", "edges", "file"])
        f.write(f"{len(chosen)}\n")
        for gi, (label, fam, p, edges, n) in enumerate(chosen):
            ldp = local_degree_profile(edges, n)
            allfeat.append(ldp)
            nbrs = [[] for _ in range(n)]
            for x, y in edges:
                nbrs[x].append(int(y)); nbrs[y].append(int(x))
            f.write(f"{n} {label}\n")
            for v in range(n):
                feat = " ".join(f"{x:.6g}" for x in ldp[v])
                f.write(f"0 {len(nbrs[v])} {' '.join(map(str, nbrs[v]))} {feat}\n".replace("  ", " "))
            w.writerow([gi, label, fam, n, len(edges), os.path.basename(p)])
    allfeat = np.concatenate(allfeat)
    print(f"\nwrote {out_txt}  ({len(chosen)} graphs, {allfeat.shape[0]} nodes, 5 LDP features/node)")
    print("LDP feature ranges (raw):  min", np.round(allfeat.min(0), 2), " max", np.round(allfeat.max(0), 2))
    print("index/label/file list  ->", os.path.join(out_dir, a.name + "_files.csv"))


if __name__ == "__main__":
    main()
