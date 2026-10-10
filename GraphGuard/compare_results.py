#!/usr/bin/env python3
"""
Compare GNNCert variants (baseline vs direction-aware) produced by dir_gnncert.py.

Reads   dir_results/<tag>/<run>/{summary.json, per_graph.npz}
Writes  dir_results/comparison/
          summary_<division>.csv / .md     mean (+- std over folds/seeds) of clean acc, certified acc @ sizes, mean certified size
          paired_<division>.csv / .md      variant - baseline per test graph: difference, 95% paired-bootstrap CI, McNemar p
          curves_<division>.png            certified accuracy vs. perturbation size  (feature / structure)
          matrices_<division>.png          certified-accuracy heat-maps + difference to baseline   (all)

"mean certified size" = average over test graphs of the number of perturbation sizes (0..NG/2-1) at which the graph is
certified correct = area under the certified-accuracy curve.
The last point of the original curves (size = NG/2) is dropped by default: with 'smallest-index wins ties' it only
contains graphs of class index 0 (e.g. 17/84 = 0.2024 in your run). Use --keep_last_size to keep it.

Usage (from GraphGuard/):
  python compare_results.py                                   # baseline = undirected_ldp5, all other tags vs it
  python compare_results.py --orig feature=new_results/GG_DS_MALNET_LDP_EPOCH_1000_SEED_42_FOLDN_3_FM_id_DAT_0_HM_md5_NG_30/GG_idx0_feature.csv \
                            --orig structure=new_results/.../GG_idx0_structure.csv
"""
import argparse, glob, json, math, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

PRETTY = {
    "undirected_ldp5": "GNNCert (undirected, LDP-5)  [baseline]",
    "directed_ldp5": "Dir-GNNCert (Dir-GIN, LDP-5)",
    "directed_dldp": "Dir-GNNCert (Dir-GIN, directed LDP-10)",
    "undirected_dldp": "GNNCert (undirected, directed LDP-10)",
}
ALL_CELLS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def load_runs(root, dataset):
    runs = []
    for sp in glob.glob(os.path.join(root, "*", "*", "summary.json")):
        s = json.load(open(sp))
        if dataset and s["dataset"] != dataset:
            continue
        d = os.path.dirname(sp)
        if not os.path.exists(os.path.join(d, "per_graph.npz")):
            continue
        s["dir"] = d
        s["z"] = np.load(os.path.join(d, "per_graph.npz"))
        runs.append(s)
    return runs


def metrics(div, NG, z, sizes):
    pred, Mp = z["pred"].astype(bool), z["Mp"]
    m = {"clean_acc": pred.astype(float)}
    if div == "all":
        for f, e in ALL_CELLS:
            m[f"cert(f={f},e={e})"] = (pred & (Mp > (f + e) * NG - f * e)).astype(float)
    else:
        for s in sizes:
            m[f"cert@{s}"] = (pred & (Mp > s)).astype(float)
        m["mean_cert_size"] = pred * np.minimum(np.maximum(np.ceil(Mp), 0), NG // 2)
    return pd.DataFrame(m, index=z["test_idx"])


def md_table(df):
    cols = list(df.columns); idx = df.index.name or ""
    out = ["| " + " | ".join([idx] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [str(r[c]) for c in cols]) + " |")
    return "\n".join(out)


def pm(vals):
    vals = np.asarray(vals, float)
    return f"{vals.mean():.4f}" if len(vals) == 1 else f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"


def mcnemar_p(b_vec, v_vec):
    from scipy.stats import binomtest
    b = int(np.sum((b_vec == 1) & (v_vec == 0))); c = int(np.sum((b_vec == 0) & (v_vec == 1)))
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="dir_results")
    ap.add_argument("--dataset", default="MALNET_DIR")
    ap.add_argument("--baseline", default="undirected_ldp5")
    ap.add_argument("--sizes", default="0,1,2,5,10,14", help="certified sizes to tabulate (feature/structure)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--orig", action="append", default=[], help="division=path/to/GG_idx0_<div>.csv of the ORIGINAL pipeline (curve only)")
    ap.add_argument("--keep_last_size", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "comparison"); os.makedirs(out, exist_ok=True)
    sizes_req = [int(x) for x in a.sizes.split(",")]
    orig = dict(o.split("=", 1) for o in a.orig)

    runs = load_runs(a.results, a.dataset)
    if not runs:
        sys.exit(f"no finished runs under {a.results}/*/*/ (need summary.json + per_graph.npz)")
    groups = defaultdict(list)                               # (division, NG, epochs) -> runs
    for r in runs:
        groups[(r["division_method"] if "division_method" in r else r["division"], r["num_group"], r["epochs"])].append(r)
    rng = np.random.default_rng(0)

    for (div, NG, E), rs in sorted(groups.items()):
        key = f"{div}_NG{NG}_E{E}"
        tags = sorted({r["tag"] for r in rs}, key=lambda t: (t != a.baseline, t))
        sizes = sorted({s for s in sizes_req if s < NG // 2} | {NG // 2 - 1})
        print(f"\n=== {a.dataset}  division={div}  NG={NG}  epochs={E}   variants: {tags}")
        per_tag_runs = {t: [r for r in rs if r["tag"] == t] for t in tags}
        per_tag_run_df = {t: [metrics(div, NG, r["z"], sizes) for r in per_tag_runs[t]] for t in tags}
        # ---------- summary table (mean ± std over runs) ----------
        rows = {}
        for t in tags:
            dfs = per_tag_run_df[t]
            rows[PRETTY.get(t, t)] = {c: pm([d[c].mean() for d in dfs]) for c in dfs[0].columns}
            rows[PRETTY.get(t, t)]["#runs"] = len(dfs)
        summ = pd.DataFrame(rows).T; summ.index.name = "variant"
        summ.to_csv(os.path.join(out, f"summary_{key}.csv")); open(os.path.join(out, f"summary_{key}.md"), "w").write(md_table(summ) + "\n")
        print(md_table(summ))
        # ---------- paired comparison against baseline ----------
        if a.baseline in tags:
            def per_graph(t):                                 # average over seeds -> one value per test graph
                return pd.concat(per_tag_run_df[t]).groupby(level=0).mean()
            base = per_graph(a.baseline); prow = []
            for t in tags:
                if t == a.baseline:
                    continue
                v = per_graph(t); ids = base.index.intersection(v.index)
                if len(ids) == 0:
                    continue
                for c in base.columns:
                    b_, v_ = base.loc[ids, c].values, v.loc[ids, c].values; d = v_ - b_
                    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)]
                    binary = set(np.unique(np.r_[b_, v_])) <= {0.0, 1.0}
                    prow.append(dict(variant=PRETTY.get(t, t), metric=c, n_graphs=len(ids), baseline=round(b_.mean(), 4),
                                     variant_value=round(v_.mean(), 4), diff=round(d.mean(), 4),
                                     ci95_low=round(float(np.percentile(bs, 2.5)), 4), ci95_high=round(float(np.percentile(bs, 97.5)), 4),
                                     mcnemar_p=round(mcnemar_p(b_, v_), 4) if binary else float("nan")))
            if prow:
                pdf = pd.DataFrame(prow); pdf.to_csv(os.path.join(out, f"paired_{key}.csv"), index=False)
                open(os.path.join(out, f"paired_{key}.md"), "w").write(md_table(pdf.set_index("variant")) + "\n")
                print("\npaired comparison vs baseline (diff = variant - baseline, 95% bootstrap CI over test graphs):")
                print(pdf.to_string(index=False))
        else:
            print(f"[note] baseline tag '{a.baseline}' has no run for this configuration -> no paired test")
        # ---------- figures ----------
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        if div != "all":
            xs = np.arange(0, NG // 2 + (1 if a.keep_last_size else 0))
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for t in tags:
                curves = np.array([[np.mean(r["z"]["pred"].astype(bool) & (r["z"]["Mp"] > s)) for s in xs] for r in per_tag_runs[t]])
                m = curves.mean(0); ax.plot(xs, m, marker="o", ms=3, label=PRETTY.get(t, t), ls="--" if t == a.baseline else "-")
                if len(curves) > 1:
                    ax.fill_between(xs, curves.min(0), curves.max(0), alpha=0.15)
            if div in orig and os.path.exists(orig[div]):
                oc = pd.read_csv(orig[div]); oc = oc[oc["size"].isin(xs)]
                ax.plot(oc["size"], oc["acc"], "k:", marker="x", label="original run (GG_idx0 csv)")
            ax.set_xlabel(f"perturbation size ({'node-feature' if div == 'feature' else 'edge'} perturbations)")
            ax.set_ylabel("certified accuracy"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7)
            ax.set_title(f"{a.dataset} – {div} division (NG={NG})"); fig.tight_layout()
            fig.savefig(os.path.join(out, f"curves_{key}.png"), dpi=200); plt.close(fig)
        else:
            R = 2 + NG // 2; Ms = {}
            for t in tags:
                mats = []
                for r in per_tag_runs[t]:
                    z = r["z"]; M = np.zeros((R, R))
                    for f in range(R):
                        for e in range(R):
                            M[f, e] = np.mean(z["pred"].astype(bool) & (z["Mp"] > (f + e) * NG - f * e))
                    mats.append(M)
                Ms[t] = np.mean(mats, 0)
            n = len(tags); fig, axs = plt.subplots(1, n + (n - 1 if a.baseline in tags else 0), figsize=(4 * (2 * n - 1 if a.baseline in tags else n), 3.8))
            axs = np.atleast_1d(axs); k = 0
            for t in tags:
                im = axs[k].imshow(Ms[t], vmin=0, vmax=1, cmap="Blues"); axs[k].set_title(PRETTY.get(t, t), fontsize=7); k += 1
                axs[k - 1].set_xlabel("edge perturbations"); axs[k - 1].set_ylabel("feature perturbations")
            if a.baseline in tags:
                for t in tags:
                    if t == a.baseline:
                        continue
                    D = Ms[t] - Ms[a.baseline]; lim = max(abs(D).max(), 1e-6)
                    axs[k].imshow(D, vmin=-lim, vmax=lim, cmap="RdBu"); axs[k].set_title("difference: " + PRETTY.get(t, t)[:28], fontsize=7)
                    axs[k].set_xlabel("edge perturbations"); k += 1
            fig.tight_layout(); fig.savefig(os.path.join(out, f"matrices_{key}.png"), dpi=200); plt.close(fig)
    print(f"\nall outputs in {out}/")


if __name__ == "__main__":
    main()
