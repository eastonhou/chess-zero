import numpy as np
from package import models, rules, mcts

class Trainer:
    def __init__(self):
        self.checkpoint_path = 'checkpoints/model.pt'
        #self.model = models.try_load_checkpoint(self.checkpoint_path).to(0)
        self.model = models.Model()
        self.optimizer = models.create_optimizer(self.model)

    def run(self):
        while True:
            train_data = self.play()
            models.update_policy(self.model, self.optimizer, train_data)
            models.save_checkpoint(self.model, self.checkpoint_path)

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
            self.print_move(len(train_data), board, move)
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

    def print_move(self, steps, board, move):
        capture = board[move[1]]
        num_red = num_black = 0
        side = 'RED' if rules.side(board[move[0]])>0 else 'BLACK'
        for piece in board:
            _side = rules.side(piece)
            if _side == 1:
                num_red += 1
            elif _side == -1:
                num_black += 1
        message = f'\r[{steps}] {side}=({move[0]},{move[1]})'\
            f' #PIECES={num_red}/{num_black}'\
            f' SCORE={rules.basic_score(board)}'
        if capture != ' ':
            message += f' CAPTURE={capture}'
        print(message, end=' '*20)
