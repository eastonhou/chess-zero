import numpy as np
from package import rules
from package.models import MoveTransform

class Node:
    def __init__(self, board, side, parent=None):
        self.board = board
        self.side = side
        self.children = None
        self.moves = None
        self.P = 0
        self.N = 0
        self.W = 0
        self.parent = parent
        self.next_to_terminal = False
        self.terminal = False

    @property
    def Q(self):
        if self.terminal:
            return 100000
        elif self.next_to_terminal:
            return -100000
        else:
            return self.W/self.N*-self.side if self.N else 0

    @property
    def U(self, c=0.5):
        return max(c*self.parent.N**0.5/(1+self.N), self.P)

    def select(self):
        scores = [x.Q+x.U for x in self.children]
        index = np.argmax(scores)
        return self.moves[index], self.children[index]

    def expand(self, moves, action_probs):
        self.moves = moves
        self.children = []
        total_P = 0
        for move in moves:
            board = rules.next_board(self.board, move)
            child = Node(board, -self.side, self)
            child.P = action_probs[MoveTransform.move_to_id(move)]
            total_P += child.P
            self.children.append(child)
        for child in self.children:
            child.P /= total_P

    def backup(self, value):
        node = self
        while node is not None:
            node.W += value
            node.N += 1
            node = node.parent

    def complete(self):
        self.parent.next_to_terminal = True
        self.terminal = True

class State:
    def __init__(self, board, side):
        self.root = Node(board, side, None)
        self.terminal = False

    def move_to_leaf(self):
        node = self.root
        while node.children is not None:
            _, node = node.select()
        return node

    def statistics(self):
        visits = [x.N if not x.terminal else 1E8 for x in self.root.children]
        probs = np.array(visits)
        probs = probs/probs.sum()
        return self.root.moves, probs

    def _make_node(self, board, side):
        return Node(board, side)

    def complete(self):
        self.terminal = True

def play(model, state):
    node = state.move_to_leaf()
    if node.terminal:
        state.complete()
        return
    if rules.gameover_position(node.board):
        value = -node.side
        node.complete()
    else:
        moves = rules.next_steps(node.board, node.side == 1)
        action_logit, value = model.forward_one(node.board, node.side)
        action_probs = action_logit.exp().detach().cpu().numpy()
        value = value.item()
        #action_probs = np.array([1/2086]*2086, dtype=np.float32)
        #value = 0
        node.expand(moves, action_probs)
    node.backup(value)

def select(moves, probs, keep):
    if keep >= 1:
        index = probs.argmax()
    else:
        probs = probs*keep+np.random.dirichlet(0.3*np.ones(probs.size))*(1-keep)
        index = np.random.choice(probs.size, p=probs)
    return moves[index]

def ponder(model, board, side, playouts=1200, keep=0.75):
    state = State(board, side)
    for _ in range(playouts):
        play(model, state)
        if state.terminal:
            break
    moves, probs = state.statistics()
    move = select(moves, probs, keep)
    action_probs = rules.MoveTransform.map_probs(moves, probs)
    return move, action_probs

if __name__ == '__main__':
    from package.models import Model
    model = Model()
    #board = 'rn ak bnr         bc   a c p p C p p             C    P P P P P                  RNBAKABNR'
    #side = -1
    #board = 'rn ak bnr         bc   a c p p C p p             C    P P P P P        B         RNBAKA NR'
    #side = 1
    board = 'rnbaka nr          c  c   bp p C p p                  P P P P P C    N           RNBAKAB R'
    side = -1
    ponder(model, board, side, keep=1)