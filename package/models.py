import torch
import torch.nn as nn
import numpy as np
from package.rules import MoveTransform, rotate_board

class Model(nn.Module):
    def __init__(self, num_residual_blocks, embedding_dim=80):
        super(__class__, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_layer = self._make_input_layer(embedding_dim)
        self.residual_blocks = self._make_residual_blocks(num_residual_blocks)
        self.policy_head = self._make_policy_head()
        self.policy_projection = self._make_policy_projection()
        self.value_head = self._make_value_head()
        self.value_projection = self._make_value_projection()
        self.embeddings = nn.Embedding(15, embedding_dim, padding_idx=0)

    def forward_one(self, board, side):
        p, v = self.forward_some([(board, side)])
        return p[0], v[0]

    def forward_some(self, batch_board_side):
        def _convert_record(board, side):
            if side == -1:
                board = rotate_board(board)
            pieces = ' rnbakcpRNBAKCP'
            board = [pieces.index(x) for x in board]
            input = np.array(board, dtype=np.int64).reshape((10,9))
            return input

        records = [_convert_record(*x) for x in batch_board_side]
        inputs = self.tensor(np.array(records))
        p, v = self.forward(inputs)
        for i,(_,side) in enumerate(batch_board_side):
            if side == -1:
                p[i] = p[i,MoveTransform.rotate_indices()]
        return p, v

    def forward(self, inputs):
        x = self.embeddings(inputs).permute(0, 3, 1, 2)
        x = self.input_layer(x)
        for m in self.residual_blocks:
            a = m(x)
            x = nn.functional.relu(a + x)
        p = self._run_head(self.policy_head, self.policy_projection, x)
        v = self._run_head(self.value_head, self.value_projection, x)
        return p, v

    def tensor(self, values):
        return torch.tensor(values, device=self.device())

    def device(self):
        return next(self.parameters()).device

    def _run_head(self, head, projection, values):
        x = head(values).reshape(values.shape[0], -1)
        x = projection(x)
        return x

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
        return nn.Linear(180, MoveTransform.action_size())

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


class Loss(nn.Module):
    def __init__(self):
        super(__class__, self).__init__()
        self.policy_criterion = nn.NLLLoss(reduction='mean')
        self.value_criterion = nn.MSELoss(reduction='mean')

    def forward(self, heads, values, target_heads, target_values):
        pass

