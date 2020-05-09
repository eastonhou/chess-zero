import os, torch
import torch.nn as nn
import numpy as np
from package.rules import MoveTransform, rotate_board

class Model(torch.jit.ScriptModule):
    def __init__(self, num_residual_blocks=7, embedding_dim=80):
        super(__class__, self).__init__()
        self.input_layer = self._make_input_layer(embedding_dim)
        self.residual_blocks = self._make_residual_blocks(num_residual_blocks)
        self.policy_head = self._make_policy_head()
        self.policy_projection = self._make_policy_projection()
        self.value_head = self._make_value_head()
        self.value_projection = self._make_value_projection()
        self.embeddings = nn.Embedding(15, embedding_dim, padding_idx=0)

    @torch.jit.script_method
    def forward(self, inputs):
        x = self.embeddings(inputs).permute(0, 3, 1, 2).contiguous()
        x = self.input_layer(x)
        for m in self.residual_blocks:
            a = m(x)
            x = nn.functional.relu(a + x)
        p = self.policy_head(x).reshape(x.shape[0], -1)
        p = self.policy_projection(p)
        v = self.value_head(x).reshape(x.shape[0], -1)
        v = self.value_projection(v).view(-1)
        v = v.view(-1)
        return p, v

    def _make_input_layer(self, embedding_dim):
        return nn.Sequential(
            nn.Conv2d(embedding_dim, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

    def _make_residual_blocks(self, num_residual_blocks):
        blocks = [self._residual_block() for _ in range(num_residual_blocks)]
        return nn.ModuleList(blocks)

    def _residual_block(self):
        return nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128))

    def _make_policy_head(self):
        return nn.Sequential(
            nn.Conv2d(128, 2, 1, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True))

    def _make_policy_projection(self):
        return nn.Sequential(
            nn.Linear(180, MoveTransform.action_size()),
            nn.LogSoftmax(dim=-1))

    def _make_value_head(self):
        return nn.Sequential(
            nn.Conv2d(128, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))

    def _make_value_projection(self):
        return nn.Sequential(
            nn.Linear(90, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
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
    #inputs = torch.zeros(1, 10, 9, dtype=torch.int64)
    #traced_module = torch.jit.trace(model, inputs)
    #traced_module.save(path)

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
        board = [pieces.index(x) for x in board]
        input = np.array(board, dtype=np.int64).reshape((10,9))
        return input
    records = [_convert_record(*x) for x in records]
    inputs = tensor(np.array(records), device)
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
