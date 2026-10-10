"""
Direction-aware GIN ("Dir-GIN"). models/graphcnn.py (original GIN) is untouched and still used for the undirected baseline.

Original GIN layer :  h' = MLP( h_v + sum_{u in N(v)} h_u )                     (neighbours of both directions are mixed)
Dir-GIN layer      :  h' = MLP( [ h_v | sum_{u->v} h_u | sum_{v->w} h_w ] )   (in- and out-neighbours kept apart)

so the model can tell callers from callees. Everything else (num_layers, MLP depth, BatchNorm+ReLU, sum graph-pooling,
layer-wise linear prediction heads summed over layers, final dropout) is copied from GraphCNN, to keep the comparison fair.
Batch format = list of objects with .node_features (n,F) and .edge_mat (2,E) where row0 = source, row1 = target.
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from mlp import MLP  # models/mlp.py (original)


class DiGraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim,
                 final_dropout, graph_pooling_type, device):
        super().__init__()
        self.num_layers, self.final_dropout, self.device = num_layers, final_dropout, device
        self.graph_pooling_type = graph_pooling_type
        self.mlps = nn.ModuleList(); self.batch_norms = nn.ModuleList()
        for layer in range(num_layers - 1):
            d_in = input_dim if layer == 0 else hidden_dim
            self.mlps.append(MLP(num_mlp_layers, 3 * d_in, hidden_dim, hidden_dim))   # [self | in | out]
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            self.linears_prediction.append(nn.Linear(input_dim if layer == 0 else hidden_dim, output_dim))

    def forward(self, batch_graph):
        dev = self.device
        X = torch.cat([g.node_features for g in batch_graph], 0).to(dev)
        sizes = [len(g.g) for g in batch_graph]
        offs = [0]
        for s in sizes:
            offs.append(offs[-1] + s)
        em = torch.cat([g.edge_mat + offs[i] for i, g in enumerate(batch_graph)], 1).to(dev)
        src, dst = em[0], em[1]
        gid = torch.repeat_interleave(torch.arange(len(batch_graph), device=dev),
                                      torch.tensor(sizes, device=dev))
        counts = torch.tensor(sizes, dtype=torch.float32, device=dev).unsqueeze(1)

        def pool(h):
            out = torch.zeros(len(batch_graph), h.shape[1], device=dev).index_add_(0, gid, h)
            return out / counts if self.graph_pooling_type == "average" else out

        h, hidden = X, [X]
        for layer in range(self.num_layers - 1):
            agg_in = torch.zeros_like(h).index_add_(0, dst, h[src])    # messages travelling along u->v arrive at v
            agg_out = torch.zeros_like(h).index_add_(0, src, h[dst])   # v collects from the nodes it points to
            h = F.relu(self.batch_norms[layer](self.mlps[layer](torch.cat([h, agg_in, agg_out], 1))))
            hidden.append(h)
        score = 0
        for layer, hl in enumerate(hidden):
            score = score + F.dropout(self.linears_prediction[layer](pool(hl)), self.final_dropout, training=self.training)
        return score
