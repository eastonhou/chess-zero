import numpy as np
from package.models import Model
from package import rules, mcts

class Trainer:
    def __init__(self):
        self.model = Model(7).to(0)

    def run(self):
        while True:
            train_data = self.play()
            self.update_policy(train_data)

    def play(self):
        board = rules.initial_board()
        side = 1
        train_data = []
        captures = []
        #restrict_round = 0
        while not rules.gameover_position(board):
            move, probs = self.ponder(board, side)
            captures.append(board[move[1]])
            train_data.append((board, side, probs))
            if len(captures)>=60 and np.all([x==' ' for x in captures[-60:]]):
                break
            board = rules.next_board(board, move)
            side *= -1
        return train_data

    def ponder(self, board, side, playouts=1200, keep=0.75):
        state = mcts.State(board, side)
        for _ in range(playouts):
            mcts.play(self.model, state)
            if state.terminal:
                break
        moves, probs = state.statistics()
        move = self.select(moves, probs, keep)
        return move, probs

    def select(self, moves, probs, keep):
        probs = probs*keep+np.random.dirichlet(0.3*np.ones(probs.size))*(1-keep)
        index = np.random.choice(probs.size, p=probs)
        return moves[index]

    def update_policy(self, train_data):
        NotImplemented
