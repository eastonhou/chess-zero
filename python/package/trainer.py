import numpy as np
from package.models import Model
from package import rules, mcts

class Trainer:
    def __init__(self):
        self.model = Model().to(0)
        self.optimizer = self.model.create_optimizer()

    def run(self):
        while True:
            train_data = self.play()
            self.model.update_policy(self.optimizer, train_data)

    def play(self, nocapture=60):
        board = rules.initial_board()
        side = 1
        train_data = []
        captures = []
        while not rules.gameover_position(board):
            move, probs = mcts.ponder(self.model, board, side)
            captures.append(board[move[1]])
            train_data.append((board, side, probs))
            if len(captures)>=nocapture and np.all([x==' ' for x in captures[-nocapture:]]):
                break
            board = rules.next_board(board, move)
            side *= -1
        if board.count('K') == 0:
            winner = -1
        elif board.count('k') == 0:
            winner = 1
        else:
            winner = 0
        train_data = [((x[0],x[1]),(x[2],winner)) for x in train_data]
        return train_data
