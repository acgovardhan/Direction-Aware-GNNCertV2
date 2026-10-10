#!/usr/bin/env python3
"""
Build the DIRECTED twin of your MALNET_LDP dataset.

It re-reads the raw MalNet-Tiny edgelists of exactly the graphs listed in
    dataset/MALNET_LDP/MALNET_LDP_files.csv
(same graphs, same order, same labels, same node renumbering) but KEEPS the caller->callee
direction. Output:
    dataset/MALNET_DIR/MALNET_DIR.txt         (same layout as MUTAG.txt, but each node row lists its SUCCESSORS only)
    dataset/MALNET_DIR/MALNET_DIR_files.csv   (copy of the file list)

        N_graphs
        n label
        0 k s_1 ... s_k        <- one row per node: dummy tag, #out-neighbours, out-neighbour ids

Because every graph/ID/label is identical to MALNET_LDP, the directed and undirected experiments are
a controlled comparison. --verify (default on) checks that, per graph, the symmetrised directed edges,
node counts, labels and the stored LDP features of MALNET_LDP are reproduced exactly.

Run from GNNCERT/GraphGuard :
    python prepare_malnet_directed.py
"""
import argparse, csv, os, sys
import numpy as np


def parse_directed(path):
    """-> (E,2) unique directed edges (src,dst) relabelled to 0..n-1, and n. Self loops dropped."""
    raw = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            a = line.split()
            if len(a) < 2:
                continue
            u, v = int(a[0]), int(a[1])
            if u != v:
                raw.append((u, v))
    if not raw:
        return np.zeros((0, 2), np.int64), 0
    e = np.array(raw, dtype=np.int64)
    ids, inv = np.unique(e.ravel(), return_inverse=True)   # same relabelling rule as prepare_malnet_ldp.py
    e = np.unique(inv.reshape(-1, 2), axis=0)               # drop duplicate directed edges only
    return e, len(ids)


def _profile(center, nbr, n, deg):
    d = deg.astype(np.float64); nd = d[nbr]
    mn = np.full(n, np.inf); np.minimum.at(mn, center, nd)
    mx = np.full(n, -np.inf); np.maximum.at(mx, center, nd)
    cnt = np.bincount(center, minlength=n).astype(np.float64)
    s1 = np.bincount(center, weights=nd, minlength=n); s2 = np.bincount(center, weights=nd * nd, minlength=n)
    safe = np.maximum(cnt, 1); mean = s1 / safe; std = np.sqrt(np.maximum(s2 / safe - mean ** 2, 0))
    z = cnt == 0
    mn[z] = mx[z] = mean[z] = std[z] = 0.0
    return np.stack([d, mn, mx, mean, std], 1)


def read_reference(path):
    """Parse MALNET_LDP.txt -> list of (n, label, set(undirected edges), feats(n,5))"""
    out = []
    with open(path) as f:
        N = int(f.readline())
        for _ in range(N):
            n, l = map(int, f.readline().split())
            es, ft = set(), []
            for j in range(n):
                r = f.readline().split(); k = int(r[1])
                for nb in r[2:2 + k]:
                    es.add((min(j, int(nb)), max(j, int(nb))))
                ft.append([float(x) for x in r[2 + k:]])
            out.append((n, l, es, np.array(ft)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/malnet-graphs-tiny")
    ap.add_argument("--ref", default="dataset/MALNET_LDP", help="folder of the undirected dataset built before")
    ap.add_argument("--ref_name", default="MALNET_LDP")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--name", default="MALNET_DIR")
    ap.add_argument("--no_verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out_dir = os.path.join(a.out, a.name); out_txt = os.path.join(out_dir, a.name + ".txt")
    if os.path.exists(out_txt) and not a.force:
        print(f"[skip] {out_txt} exists (use --force)"); return

    ref_csv = os.path.join(a.ref, a.ref_name + "_files.csv")
    rows = list(csv.DictReader(open(ref_csv)))
    ref = None if a.no_verify else read_reference(os.path.join(a.ref, a.ref_name + ".txt"))
    if ref is not None and len(ref) != len(rows):
        sys.exit(f"{ref_csv} lists {len(rows)} graphs but {a.ref_name}.txt has {len(ref)}: the files are out of sync "
                 f"(was MALNET_LDP rebuilt after the csv was written?).")

    os.makedirs(out_dir, exist_ok=True)
    recip, fam_recip = [], {}
    with open(out_txt, "w") as f:
        f.write(f"{len(rows)}\n")
        for gi, r in enumerate(rows):
            p = os.path.join(a.src, r["family"], r["file"])
            e, n = parse_directed(p)
            if ref is not None:
                rn, rl, res, rft = ref[gi]
                sym = set(zip(np.minimum(e[:, 0], e[:, 1]).tolist(), np.maximum(e[:, 0], e[:, 1]).tolist()))
                assert n == rn, f"graph {gi} ({r['file']}): {n} nodes here vs {rn} in {a.ref_name}"
                assert int(r["label"]) == rl, f"graph {gi}: label differs"
                assert sym == res, f"graph {gi} ({r['file']}): undirected edge sets differ"
                # LDP columns stored in MALNET_LDP == LDP recomputed from the symmetrised directed edges
                u = np.concatenate([np.array(sorted(sym))[:, 0], np.array(sorted(sym))[:, 1]])
                v = np.concatenate([np.array(sorted(sym))[:, 1], np.array(sorted(sym))[:, 0]])
                ldp = _profile(u, v, n, np.bincount(u, minlength=n))
                assert np.allclose(ldp, rft, rtol=1e-4, atol=1e-4), f"graph {gi}: LDP mismatch"
            es = set(map(tuple, e.tolist()))
            rec = sum((d, s) in es for s, d in es) / max(len(es), 1)
            recip.append(rec); fam_recip.setdefault(r["family"], []).append(rec)
            succ = [[] for _ in range(n)]
            for s, d in e:
                succ[s].append(int(d))
            f.write(f"{n} {r['label']}\n")
            for v_ in range(n):
                f.write(f"0 {len(succ[v_])} {' '.join(map(str, succ[v_]))}".rstrip() + "\n")
    import shutil; shutil.copy(ref_csv, os.path.join(out_dir, a.name + "_files.csv"))
    print(f"wrote {out_txt}  ({len(rows)} graphs)")
    if ref is not None:
        print("VERIFIED: node counts, undirected edge sets and LDP features are identical to", a.ref_name)
    print("share of directed edges that also exist in the reverse direction (reciprocity):")
    for fam, v in fam_recip.items():
        print(f"   {fam:12s} mean {np.mean(v):.3f}")
    print(f"   overall      mean {np.mean(recip):.3f}   (near 0 => direction carries information)")


if __name__ == "__main__":
    main()
