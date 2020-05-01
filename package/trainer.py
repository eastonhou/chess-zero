import numpy as np
from package.models import Model
from package import rules, mcts

class Trainer:
    def __init__(self):
        self.model = Model(7).to(0)
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
            move, probs = self.ponder(board, side)
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

    def ponder(self, board, side, playouts=1200, keep=0.75):
        state = mcts.State(board, side)
        for _ in range(playouts):
            mcts.play(self.model, state)
            if state.terminal:
                break
        moves, probs = state.statistics()
        move = self.select(moves, probs, keep)
        probs = rules.MoveTransform.map_probs(moves, probs)
        return move, probs

    def select(self, moves, probs, keep):
        probs = probs*keep+np.random.dirichlet(0.3*np.ones(probs.size))*(1-keep)
        index = np.random.choice(probs.size, p=probs)
        return moves[index]
