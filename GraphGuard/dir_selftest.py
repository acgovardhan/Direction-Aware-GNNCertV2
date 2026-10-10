#!/usr/bin/env python3
"""Quick sanity checks (no dataset needed). Run from GraphGuard/:  python dir_selftest.py"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dir_data import DGraph, node_features, prepare_groups, make_views, build_batch
from dir_model import DiGraphCNN
from models.graphcnn import GraphCNN

def mk(src, dst, n, feat="ldp5"):
    s, d = np.array(src), np.array(dst)
    return DGraph(n, 0, s, d, node_features(s, d, n, feat))

# 1) direction sensitivity: a call graph and its reverse
out_star = mk([0, 0, 0, 0, 1], [1, 2, 3, 4, 5], 6, "dldp")           # hub 0 calls 1..4, 1 calls 5
in_star = mk([1, 2, 3, 4, 5], [0, 0, 0, 0, 1], 6, "dldp")            # same edges reversed
for g in (out_star, in_star):
    g.x = torch.log1p(torch.rand(g.n, 10)) * 0 + g.x
G = [out_star, in_star]; prepare_groups(G, 4)
torch.manual_seed(0); dev = torch.device("cpu")
dm = DiGraphCNN(3, 2, 10, 16, 3, 0.0, "sum", dev).eval()
um = GraphCNN(3, 2, 10, 16, 3, 0.0, False, "sum", "sum", dev).eval()
bd = build_batch(G, [(0, -1, -1), (1, -1, -1)], "directed", -99.0)
bu = build_batch(G, [(0, -1, -1), (1, -1, -1)], "undirected", -99.0)
# use identical (direction-free) node features so ONLY the edge direction differs
x = torch.rand(6, 10)
for b in (bd, bu):
    for it in b: it.node_features = x.clone()
with torch.no_grad():
    d = dm(bd); u = um(bu)
print("Dir-GIN   logits differ between G and reversed G :", not torch.allclose(d[0], d[1], atol=1e-6))
print("orig GIN  logits identical for G and reversed G   :", torch.allclose(u[0], u[1], atol=1e-5))
assert not torch.allclose(d[0], d[1], atol=1e-6) and torch.allclose(u[0], u[1], atol=1e-5)

# 2) ordered-edge hashing: u->v and v->u are different hash inputs; every edge lands in exactly one group
g = mk([0, 1, 2, 3], [1, 0, 3, 2], 4); prepare_groups([g], 5)
assert sum(len(s) for s in g.sel["directed"]) == 4 and sum(len(s) for s in g.sel["undirected"]) == 2
print("structure division partitions the edge set (directed: 4 edges, undirected: 2 edges)  OK")
# 3) 'all' view count and feature-mask partition
assert len(make_views([0, 1], "all", 4)) == 2 * 16 and len(make_views([0], "feature", 30)) == 30
print("self-test passed")
