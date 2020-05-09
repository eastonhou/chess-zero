import os, torch
import torch.nn as nn
import numpy as np
from package.rules import MoveTransform, rotate_board
from torch_geometric import nn as gnn

class Model(torch.jit.ScriptModule):
    def __init__(self, num_residual_blocks=7, embedding_dim=120, node_features=768, layers=5):
        super(__class__, self).__init__()
        self.piece_embeddings = nn.Embedding(15, embedding_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(90, embedding_dim)
        self.mode_models = nn.Sequential(
            gnn.FeaStConv(embedding_dim*2, node_features, 8),
            *[gnn.FeaStConv(node_features, node_features, 8) for _ in range(layers)])
        self.policy_projection = self._make_policy_projection(node_features)
        self.value_projection = self._make_value_projection(node_features)

    #@torch.jit.script_method
    def forward(self, x, pos, graphs):
        edge_index = self._make_edge_index(graphs)
        piece_features = self.piece_embeddings(x)
        position_features = self.position_embeddings(pos)
        node_features = torch.cat([piece_features, position_features], dim=-1)
        node_features = self.node_models(node_features, edge_index)
        states = self._group_states(node_features, graphs)
        p = self.policy_projection(states)
        v = self.value_projection(states)
        return p, v

    def _make_edge_index(self, graphs):
        offset = 0
        edges = set()
        for graph in graphs:
            for i,j in utils.enumerate_pairs(len(graph)):
                edges.add((i+offset,j+offset))
                edges.add((j+offset,i+offset))
            offset += len(graph)
        rows, cols = zip(*edges)
        rows, cols = np.array(rows), np.array(cols)
        return torch.tensor(rows), torch.tensor(cols)

    def _group_states(self, x, graphs):
        offset = 0
        states = []
        for graph in graphs:
            values = x[offset:offset+len(graph)]
            states.append(values.sum(dim=0))
            offset += len(graph)
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
    p, v = model(inputs)
    for i,(_,side) in enumerate(batch_board_side):
        if side == -1:
            p[i] = p[i,MoveTransform.rotate_indices()]
            v[i] = -v[i]
    return p, v

def save_checkpoint(model, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    model.save(path)

def try_load_checkpoint(path):
    if os.path.isfile(path):
        return torch.jit.load(path)
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
    inputs = torch.tensor(x, device), torch.tensor(pos, device), graphs
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
        p, v = model(inputs)
        tp, tv = tensor(tp, device), tensor(tv, device).float()
        ploss = (-p*tp).sum()
        vloss = nn.functional.mse_loss(v, tv)
        loss = ploss + vloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'LOSS: {loss.div(inputs.shape[0]).item()}')

def create_optimizer(model):
    return torch.optim.AdamW(model.parameters())

def tensor(values, device):
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    return torch.tensor(values, device=device)

def model_device(model):
    return next(model.parameters()).device
