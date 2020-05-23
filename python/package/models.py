import os, torch
import torch.nn as nn
import numpy as np
from package.rules import MoveTransform, rotate_board
from package import utils
from torch_geometric import nn as gnn

class EdgeModel(nn.Module):
    def __init__(self, node_features, in_features, out_features, dropout):
        super(__class__, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh())
        self.mlp = nn.Sequential(
            nn.Linear(out_features+node_features*2, node_features),
            gnn.BatchNorm(node_features, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(node_features, out_features),
            nn.ReLU(inplace=True))

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.input_layer(edge_attr)
        row, col = edge_index
        x = torch.cat([x[row], x[col], edge_attr], 1)
        y = self.mlp(x)
        return y

class Model(nn.Module):
    def __init__(self, layers=5):
        super(__class__, self).__init__()
        self.piece_dim = 128
        self.position_dim = 128
        self.edge_dim = 128
        self.node_features = self.piece_dim+self.position_dim
        self.piece_embeddings = nn.Embedding(15, self.piece_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(90, self.position_dim)
        self.edge_embeddings = nn.Embedding(90*90, self.edge_dim)
        self.layers =  nn.ModuleList([self._make_node_net() for _ in range(layers)])
        self.policy_projection = self._make_policy_projection(self.node_features)
        self.value_projection = self._make_value_projection(self.node_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, pos, graphs):
        edge_index, edge_ids = self._make_edge_index(pos, graphs)
        edge_attr = self.edge_embeddings(edge_ids.long())
        piece_features = self.piece_embeddings(x)
        position_features = self.position_embeddings(pos)
        x = torch.cat([piece_features, position_features], dim=-1)
        for edge_model,edge_norm,node_model,node_norm in self.layers:
            edge_attr = edge_model(x, edge_index, edge_attr)
            edge_attr = edge_norm(edge_attr)
            x = node_model(x, edge_index, edge_attr)
            x = self.dropout(x)
            x = node_norm(x)
        states = self._group_states(x, graphs)
        p = self.policy_projection(states)
        v = self.value_projection(states)
        return p, v.view(-1)

    def tensor(self, value):
        device = next(self.parameters()).device
        return torch.tensor(value, device=device)

    def _make_node_net(self):
        return nn.ModuleList([
            EdgeModel(self.node_features, self.edge_dim, self.edge_dim, 0.1),
            gnn.BatchNorm(self.edge_dim, momentum=0.01),
            gnn.CGConv(self.node_features, self.edge_dim),
            gnn.BatchNorm(self.node_features, momentum=0.01)])

    def _make_edge_index(self, pos, graphs):
        offset = 0
        edges = []
        ida, idb = [], []
        for graph in graphs:
            for i,j in utils.enumerate_pairs(graph):
                edges.append((i+offset,j+offset))
                edges.append((j+offset,i+offset))
                ida.append(i+offset)
                idb.append(j+offset)
                #ids.append(pos[i+offset]*90+pos[j+offset])
                #ids.append(pos[j+offset]*90+pos[i+offset])
            offset += graph
        indices = np.array(edges).transpose()
        ida, idb = pos[ida], pos[idb]
        ids = torch.stack([ida*90+idb,ida+idb*90],dim=-1)
        return self.tensor(indices).long(), ids.view(-1).long()

    def _group_states(self, x, graphs):
        offset = 0
        states = []
        for graph in graphs:
            values = x[offset:offset+graph]
            states.append(values.sum(dim=0))
            offset += graph
        return torch.stack(states, dim=0)

    def _make_policy_projection(self, node_features):
        return nn.Sequential(
            nn.Linear(node_features, MoveTransform.action_size()),
            nn.LogSoftmax(dim=-1))

    def _make_value_projection(self, node_features):
        return nn.Sequential(
            nn.Linear(node_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh())

def forward_one(model, board, side):
    p, v = forward_some(model, [(board, side)])
    return p[0], v[0]

def forward_some(model, batch_board_side):
    inputs = convert_inputs(batch_board_side, model_device(model))
    p, v = model(*inputs)
    for i,(_,side) in enumerate(batch_board_side):
        if side == -1:
            p[i] = p[i,MoveTransform.rotate_indices()]
            v[i] = -v[i]
    return p, v

def save_checkpoint(model, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(model, path)

def try_load_checkpoint(path):
    if os.path.isfile(path):
        return torch.load(path)
    else:
        model = Model()
        save_checkpoint(model, path)
        return model

def convert_inputs(records, device):
    def _convert_record(board, side):
        if side == -1:
            board = rotate_board(board)
        pieces = ' rnbakcpRNBAKCP'
        x, pos = [], []
        for i,_x in enumerate(pieces):
            _x = pieces.index(_x)
            if _x != 0:
                x.append(_x)
                pos.append(i)
        return x, pos
    x, pos, graphs = [], [], []
    for board,side in records:
        _x, _pos = _convert_record(board, side)
        x += _x
        pos += _pos
        graphs.append(len(_x))
    inputs = torch.tensor(x, device=device), torch.tensor(pos, device=device), graphs
    return inputs

def convert_targets(targets, sides):
    ps, vs = [], []
    for (p,v),s in zip(targets, sides):
        if s == -1:
            p = p[MoveTransform.rotate_indices()]
            v = -v
        ps.append(p)
        vs.append(v)
    return ps, vs

def update_policy(model, optimizer, train_data, epochs=10):
    device = model_device(model)
    records, targets = zip(*train_data)
    inputs = convert_inputs(records, device)
    tp, tv = convert_targets(targets, [x[1] for x in records])
    for _ in range(epochs):
        p, v = model(*inputs)
        tp, tv = tensor(tp, device), tensor(tv, device).float()
        ploss = (-p*tp).sum()
        vloss = nn.functional.mse_loss(v, tv)
        loss = ploss + vloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'LOSS: {loss.div(len(inputs[-1])).item()}')

def create_optimizer(model):
    return torch.optim.AdamW(model.parameters())

def tensor(values, device):
    if torch.is_tensor(values):
        return values.to(device)
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    return torch.tensor(values, device=device)

def model_device(model):
    return next(model.parameters()).device
