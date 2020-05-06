import numpy as np
from package import rules, utils, models

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

    def select_to_leaf(self):
        node = self
        while node.children is not None:
            _, node = node.select()
        return node

    def select_multiple(self, n):
        nodes = []
        for _ in range(n):
            node = self.select_to_leaf()
            if node in nodes:
                break
            node.backup(node.side*1000)
            nodes.append(node)
            if node.terminal:
                break
        for node in nodes:
            node.backup(node.side*1000, -1)
        return nodes

    def expand(self, moves, action_probs):
        self.moves = moves
        self.children = []
        total_P = 0
        for move in moves:
            board = rules.next_board(self.board, move)
            child = Node(board, -self.side, self)
            child.P = action_probs[rules.MoveTransform.move_to_id(move)]
            total_P += child.P
            self.children.append(child)
        for child in self.children:
            child.P /= total_P

    def backup(self, value, direction=1):
        node = self
        while node is not None:
            node.W += value*direction
            node.N += 1*direction
            node = node.parent

    def complete(self):
        self.parent.next_to_terminal = True
        self.terminal = True

class State:
    def __init__(self, board, side):
        self.root = Node(board, side, None)
        self.terminal = False

    def statistics(self):
        visits = [x.N if not x.terminal else 1E8 for x in self.root.children]
        probs = np.array(visits)
        probs = probs/probs.sum()
        return self.root.moves, probs

    def complete(self):
        self.terminal = True

def play(model, state):
    node = state.root.select_to_leaf()
    if node.terminal:
        state.complete()
        return
    if rules.gameover_position(node.board):
        value = -node.side
        node.complete()
    else:
        moves = rules.next_steps(node.board, node.side==1)
        action_logit, value = model.forward_one(node.board, node.side)
        action_probs = action_logit.exp().detach().cpu().numpy()
        value = value.item()
        node.expand(moves, action_probs)
    node.backup(value)

def play_multiple(model, state, n):
    timer = utils.Timer()
    nodes = state.root.select_multiple(n)
    nonterminals = []
    for node in nodes:
        if rules.gameover_position(node.board):
            node.complete()
            node.backup(-node.side)
        else:
            nonterminals.append(node)
    if not nonterminals:
        return
    timer.check('-select')
    records = [(x.board, x.side) for x in nonterminals]
    logits, values = models.forward_some(model, records)
    probs = logits.exp().detach().cpu().numpy()
    values = values.detach().cpu().numpy()
    timer.check('-model')
    moves = [rules.next_steps(x.board, x.side==1) for x in nonterminals]
    timer.check('-moves')
    for _node, _moves, _probs, _value in zip(nonterminals, moves, probs, values):
        _node.expand(_moves, _probs)
        _node.backup(_value)
    timer.check('-expand')

def select(moves, probs, keep):
    if keep >= 1:
        index = probs.argmax()
    else:
        probs = probs*keep+np.random.dirichlet(0.3*np.ones(probs.size))*(1-keep)
        index = np.random.choice(probs.size, p=probs)
    return moves[index]

def ponder(model, board, side, playouts=200, keep=0.75):
    state = State(board, side)
    timer = utils.Timer()
    for _ in range(playouts):
        play_multiple(model, state, 64)
        timer.check('ponder.step', count=1)
        if state.terminal:
            break
    timer.print()
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