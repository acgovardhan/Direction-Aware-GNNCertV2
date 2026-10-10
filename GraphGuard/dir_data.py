"""
Data layer of the direction-aware GNNCert pipeline (does NOT touch util.py / division.py).

* load_directed_dataset(): reads dataset/<NAME>/<NAME>.txt (successor lists) -> list[DGraph]
* node features  : 'ldp5' = LDP of the symmetrised graph (identical to MALNET_LDP, log1p-scaled)
                   'dldp' = [LDP(G) | LDP(G^T)]  = out-profile (10 dims total, log1p-scaled):
                            out-degree + min/max/mean/std of successors' out-degree,
                            in-degree  + min/max/mean/std of predecessors' in-degree
* division       : feature / structure / all, exactly as in GNNCert (md5(str(x)) % num_group), but
                   - 'directed'   : an edge is the ORDERED pair "src;dst"  (so u->v and v->u can land in different groups)
                   - 'undirected' : an edge is the canonical pair "min;max" (what the original code hashes)
  Subgraphs are built lazily (View) so NUM_GROUP**2 subgraphs per graph do not blow up memory.
"""
import hashlib
import numpy as np
import torch

HASHES = {
    "md5": lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16),
    "sha256": lambda x: int(hashlib.sha256(str(x).encode()).hexdigest(), 16),
    "sha512": lambda x: int(hashlib.sha512(str(x).encode()).hexdigest(), 16),
}


def _profile(center, nbr, n, deg):
    """Local-degree-profile: (deg, min, max, mean, std of neighbours' deg). neighbour of center[i] is nbr[i]."""
    d = deg.astype(np.float64); nd = d[nbr]
    mn = np.full(n, np.inf); np.minimum.at(mn, center, nd)
    mx = np.full(n, -np.inf); np.maximum.at(mx, center, nd)
    cnt = np.bincount(center, minlength=n).astype(np.float64)
    s1 = np.bincount(center, weights=nd, minlength=n); s2 = np.bincount(center, weights=nd * nd, minlength=n)
    safe = np.maximum(cnt, 1); mean = s1 / safe; std = np.sqrt(np.maximum(s2 / safe - mean ** 2, 0))
    z = cnt == 0
    mn[z] = mx[z] = mean[z] = std[z] = 0.0
    return np.stack([d, mn, mx, mean, std], 1)


def node_features(src, dst, n, feature_set):
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    und = np.unique(np.stack([lo, hi], 1), axis=0) if len(src) else np.zeros((0, 2), np.int64)
    u = np.concatenate([und[:, 0], und[:, 1]]); v = np.concatenate([und[:, 1], und[:, 0]])
    ldp = _profile(u, v, n, np.bincount(u, minlength=n))
    if feature_set == "ldp5":
        f = ldp
    elif feature_set == "dldp":
        out_p = _profile(src, dst, n, np.bincount(src, minlength=n))   # successors' out-degree
        in_p = _profile(dst, src, n, np.bincount(dst, minlength=n))    # predecessors' in-degree
        f = np.concatenate([out_p, in_p], 1)
    else:
        raise ValueError(feature_set)
    return torch.tensor(np.log1p(f), dtype=torch.float32)               # log1p tames heavy-tailed degrees


class DGraph:
    """One call graph. Edge arrays are stored for BOTH modes; groups are attached by prepare_groups()."""
    def __init__(self, n, label, src, dst, x):
        self.n, self.label, self.x = n, label, x
        self.src, self.dst = src, dst                                    # directed, unique, no self loops
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        und = np.unique(np.stack([lo, hi], 1), axis=0) if len(src) else np.zeros((0, 2), np.int64)
        self.lo, self.hi = und[:, 0].copy(), und[:, 1].copy()            # canonical undirected, unique


def load_directed_dataset(name, feature_set="ldp5", root="dataset"):
    graphs, label_dict = [], {}
    with open(f"{root}/{name}/{name}.txt") as f:
        N = int(f.readline())
        for _ in range(N):
            n, l = [int(w) for w in f.readline().split()]
            label_dict.setdefault(l, len(label_dict))                    # same first-appearance mapping as util.load_data
            s, d = [], []
            for j in range(n):
                row = f.readline().split(); k = int(row[1])
                for nb in row[2:2 + k]:
                    s.append(j); d.append(int(nb))
            s = np.array(s, dtype=np.int64); d = np.array(d, dtype=np.int64)
            graphs.append(DGraph(n, label_dict[l], s, d, node_features(s, d, n, feature_set)))
    return graphs, len(label_dict), label_dict


def prepare_groups(graphs, num_group, hash_name="md5"):
    """Attach node groups (feature division) and edge groups (structure division) for both modes."""
    h = HASHES[hash_name]
    cache = {}
    def nh(i):
        if i not in cache:
            cache[i] = h(i) % num_group
        return cache[i]
    for g in graphs:
        g.node_group = np.array([nh(i) for i in range(g.n)], dtype=np.int64)
        g.node_group_t = torch.from_numpy(g.node_group)
        g.eg_dir = np.array([h(f"{s};{d}") % num_group for s, d in zip(g.src.tolist(), g.dst.tolist())], dtype=np.int64)
        g.eg_und = np.array([h(f"{s};{d}") % num_group for s, d in zip(g.lo.tolist(), g.hi.tolist())], dtype=np.int64)
        g.sel = {}
        for mode, eg in (("directed", g.eg_dir), ("undirected", g.eg_und)):
            g.sel[mode] = [np.nonzero(eg == i)[0] for i in range(num_group)]


class Item:
    """Minimal stand-in for util.S2VGraph: exactly the attributes models/graphcnn.py reads."""
    __slots__ = ("g", "node_features", "edge_mat", "label", "neighbors", "max_neighbor")


def make_views(graph_ids, division, num_group):
    """(graph_index, structure_group, feature_group); -1 = not divided along that axis. Order as in division.py."""
    v = []
    for gi in graph_ids:
        if division == "feature":
            v += [(gi, -1, j) for j in range(num_group)]
        elif division == "structure":
            v += [(gi, i, -1) for i in range(num_group)]
        elif division == "all":
            v += [(gi, i, j) for i in range(num_group) for j in range(num_group)]
        else:
            raise ValueError(division)
    return v


def build_item(graphs, view, direction, mask_value):
    gi, si, fj = view
    g = graphs[gi]
    if direction == "directed":
        s_all, d_all = g.src, g.dst
    else:
        s_all, d_all = g.lo, g.hi
    if si >= 0:
        sel = g.sel[direction][si]
        s, d = s_all[sel], d_all[sel]
    else:
        s, d = s_all, d_all
    x = g.x
    if fj >= 0:
        x = x.clone()
        x[g.node_group_t != fj] = mask_value
    if direction == "directed":
        em = np.stack([s, d]) if len(s) else np.zeros((2, 0), np.int64)           # row0=src, row1=dst
    else:
        em = np.stack([np.concatenate([s, d]), np.concatenate([d, s])]) if len(s) else np.zeros((2, 0), np.int64)
    it = Item()
    it.g = range(g.n); it.node_features = x; it.edge_mat = torch.from_numpy(em).long()
    it.label = g.label; it.neighbors = []; it.max_neighbor = 0
    return it


def build_batch(graphs, views, direction, mask_value):
    return [build_item(graphs, v, direction, mask_value) for v in views]
