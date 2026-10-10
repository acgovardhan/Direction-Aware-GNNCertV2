#!/usr/bin/env python3
"""
Idempotent patch for GNNCERT/GraphGuard (backs up originals as *.orig). Run from GraphGuard/ :
    python patch_gnncert.py              # all patches
    python patch_gnncert.py --no-division-fix
"""
import argparse, os, shutil

def edit(path, old, new, tag):
    s = open(path).read()
    if tag in s:
        print(f"[ok]   {path}: already patched ({tag})"); return
    if old not in s:
        print(f"[WARN] {path}: pattern not found for {tag} - edit by hand"); return
    if not os.path.exists(path + ".orig"):
        shutil.copy(path, path + ".orig")
    open(path, "w").write(s.replace(old, new))
    print(f"[done] {path}: {tag}")

ap = argparse.ArgumentParser(); ap.add_argument("--no-division-fix", action="store_true")
a = ap.parse_args()

# 1) util.load_dataset -> route MALNET* to the LDP loader
edit("util.py",
     "def load_dataset(dataset, degree_as_tag):\n",
     "def load_dataset(dataset, degree_as_tag):\n"
     "    if dataset.upper().startswith('MALNET'):  # MALNET_LDP_PATCH\n"
     "        from malnet_loader import load_data_ldp\n"
     "        return load_data_ldp(dataset, degree_as_tag)\n",
     "MALNET_LDP_PATCH")

# 2) division.py: copy() is shallow, so `subgraph.node_features[mask] = DEFAULT` edited ONE shared tensor
if not a.no_division_fix:
    edit("division.py",
         "        subgraph = copy(graph)\n        subgraph.node_features[group!=i] = DEFAULT\n        subgraphs.append(subgraph)",
         "        subgraph = copy(graph)\n        subgraph.node_features = graph.node_features.clone()  # DIVFIX\n"
         "        subgraph.node_features[group!=i] = DEFAULT\n        subgraphs.append(subgraph)",
         "DIVFIX")
    edit("division.py",
         "            subgraph = copy(graph)\n            subgraph.node_features[group != i] = DEFAULT\n            subgraphs.append(subgraph)",
         "            subgraph = copy(graph)\n            subgraph.node_features = graph.node_features.clone()  # DIVFIX2\n"
         "            subgraph.node_features[group != i] = DEFAULT\n            subgraphs.append(subgraph)",
         "DIVFIX2")

# 3) main_graphguard.py: evaluation every 2 epochs over ALL divided test graphs is very slow for big graphs
edit("main_graphguard.py",
     "        if epoch % 2 == 0:\n            acc_train, acc_test = test(",
     "        if epoch % int(__import__('os').environ.get('EVAL_EVERY', '2')) == 0:  # EVALPATCH\n            acc_train, acc_test = test(",
     "EVALPATCH")
