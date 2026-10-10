"""
Loader for the LDP-featured MalNet file written by prepare_malnet_ldp.py.
Returns exactly what util.load_data returns: (list[S2VGraph], num_classes, tag2index)
but node_features are the 5 LDP values (log1p-scaled) instead of a one-hot tag.
"""
import networkx as nx
import numpy as np
import torch
from util import S2VGraph


def load_data_ldp(dataset, degree_as_tag=False):
    if degree_as_tag:
        print("[malnet_loader] degree_as_tag ignored: LDP already contains the degree.")
    g_list, label_dict = [], {}
    with open(f"dataset/{dataset}/{dataset}.txt", "r") as f:
        n_g = int(f.readline().strip())
        for _ in range(n_g):
            n, l = [int(w) for w in f.readline().strip().split()]
            label_dict.setdefault(l, len(label_dict))
            g = nx.Graph()
            feats = []
            for j in range(n):
                g.add_node(j)
            for j in range(n):
                row = f.readline().strip().split()
                k = int(row[1])
                for nb in row[2:2 + k]:
                    g.add_edge(j, int(nb))
                feats.append([float(x) for x in row[2 + k:]])
            assert len(g) == n
            sg = S2VGraph(g, l, [0] * n)
            # log1p tames the heavy-tailed degrees (hundreds) so sum-aggregation does not explode
            sg.node_features = torch.log1p(torch.tensor(np.array(feats), dtype=torch.float32))
            g_list.append(sg)

    for sg in g_list:
        sg.neighbors = [list(sg.g.neighbors(i)) for i in range(len(sg.g))]
        sg.max_neighbor = max(len(x) for x in sg.neighbors)
        sg.label = label_dict[sg.label]
        edges = [list(p) for p in sg.g.edges()]
        edges.extend([[j, i] for i, j in edges])
        sg.edge_mat = torch.LongTensor(edges).transpose(0, 1)
    return g_list, len(label_dict), {0: 0}
