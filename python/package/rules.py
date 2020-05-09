import numpy as np
BASE = 100
GAMEOVER_THRESHOLD = 150 * BASE

def next_steps(board, red):
    steps = []
    for i in range(len(board)):
        chess = board[i]
        if side(chess) == 0 or is_red(chess) != red:
            continue
        moves = next_chess_steps[chess](board, i)
        steps += [(i, m) for m in moves]
    return steps

def can_move(board, i0, i1):
    moves = next_chess_steps[board[i0]](board, i0)
    return i1 in moves

def is_red(chess):
    return 'A' <= chess <= 'Z'


def position_1(x, y):
    return x + y * 9


def position_2(pos):
    return pos % 9, pos // 9


def flip_side(chess):
    if is_red(chess):
        return chess.lower()
    elif ' ' == chess:
        return chess
    else:
        return chess.upper()


def rotate_board(board):
    board = board[::-1]
    board = ''.join([flip_side(x) for x in board])
    return board

def valid_position2(x, y):
    return x >= 0 and x < 9 and y >= 0 and y < 10


def side(chess):
    if is_red(chess): return 1
    elif ' ' == chess: return 0
    else: return -1


def gameover_position(board):
    return board.upper().count('K') < 2


def next_rider_steps(board, pos):
    steps = []
    px, py = position_2(pos)
    for r in [range(px+1, 9), range(px-1, -1, -1)]:
        for x in r:
            p = position_1(x, py)
            if side(board[p]) * side(board[pos]) <= 0:
                steps.append(p)
            if ' ' != board[p]:
                break
    for r in [range(py+1, 10), range(py-1, -1, -1)]:
        for y in r:
            p = position_1(px, y)
            if side(board[p]) * side(board[pos]) <= 0:
                steps.append(p)
            if ' ' != board[p]:
                break
    return steps


def next_horse_steps(board, pos):
    steps = []
    px, py = position_2(pos)
    for dx, dy in [(-2,-1),(-2,1),(2,-1),(2,1),(-1,-2),(-1,2),(1,-2),(1,2)]:
        if valid_position2(px+dx, py+dy) is False:
            continue
        bx = int(dx / 2)
        by = int(dy / 2)
        if board[position_1(px+bx, py+by)] != ' ':
            continue
        p = position_1(px+dx,py+dy)
        if side(board[p]) * side(board[pos]) <= 0:
            steps.append(p)
    return steps


def next_elephant_steps(board, pos):
    steps = []
    px, py = position_2(pos)
    for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2)]:
        if valid_position2(px+dx, py+dy) is False:
            continue
        bx = dx // 2
        by = dy // 2
        if board[position_1(px+bx, py+by)] != ' ':
            continue
        if py+dy not in [0, 2, 4, 5, 7, 9]:
            continue
        p = position_1(px+dx,py+dy)
        if side(board[p]) * side(board[pos]) <= 0:
            steps.append(p)
    return steps


def next_bishop_steps(board, pos):
    steps = []
    px, py = position_2(pos)
    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        if valid_position2(px+dx,py+dy) is False:
            continue
        if px+dx not in [3,4,5] or py+dy not in [0,1,2,7,8,9]:
            continue
        p = position_1(px+dx,py+dy)
        if side(board[p]) * side(board[pos]) <= 0:
            steps.append(p)
    return steps


def next_king_steps(board, pos):
    steps = []
    px, py = position_2(pos)
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        if valid_position2(px+dx,py+dy) is False:
            continue
        if px+dx not in [3,4,5] or py+dy not in [0,1,2,7,8,9]:
            continue
        p = position_1(px+dx,py+dy)
        if side(board[p]) * side(board[pos]) <= 0:
            steps.append(p)
    if py <= 2:
        rng = range(py+1, 10)
    else:
        rng = range(py-1, -1, -1)
    for y in rng:
        p = position_1(px, y)
        if board[p] != ' ':
            if board[p] in ['k', 'K']:
                steps.append(p)
            break
    return steps


def next_cannon_steps(board, pos):
    steps = []
    px, py = position_2(pos)
    for r in [range(px+1, 9), range(px-1, -1, -1)]:
        counter = 0
        for x in r:
            p = position_1(x, py)
            if counter == 0:
                if ' ' == board[p]:
                    steps.append(p)
                else:
                    counter += 1
            elif counter == 1:
                if side(board[p]) * side(board[pos]) < 0:
                    steps.append(p)
                if ' ' != board[p]:
                    counter = 2
                    break
            else:
                break
    for r in [range(py+1, 10), range(py-1, -1, -1)]:
        counter = 0
        for y in r:
            p = position_1(px, y)
            if counter == 0:
                if ' ' == board[p]:
                    steps.append(p)
                else:
                    counter += 1
            elif counter == 1:
                if side(board[p]) * side(board[pos]) < 0:
                    steps.append(p)
                if ' ' != board[p]:
                    counter = 2
                    break
            else:
                break
    return steps


def next_pawn_steps(board, pos):
    steps = []
    px, py = position_2(pos)
    red_king_pos = find_chess(board, 'K')[0]
    if is_red(board[pos]) == (red_king_pos >= 45):
        if py <= 4:
            possible = [(0,-1),(-1,0),(1,0)]
        else:
            possible = [(0,-1)]
    else:
        if py >= 5:
            possible = [(0,1),(-1,0),(1,0)]
        else:
            possible = [(0,1)]
    for dx, dy in possible:
        if valid_position2(px+dx,py+dy) is False:
            continue
        p = position_1(px+dx, py+dy)
        if side(board[p]) * side(board[pos]) <= 0:
            steps.append(p)
    return steps


def find_chess(board, chess):
    return [i for i in range(90) if board[i] == chess]


def basic_score(board):
    return sum([score_map[c] for c in board])


def initial_board():
    return 'rnbakabnr##########c#####c#p#p#p#p#p##################P#P#P#P#P#C#####C##########RNBAKABNR'.replace('#', ' ')


def next_board(board, move):
    r = [b for b in board]
    f, t = move
    r[t] = r[f]
    r[f] = ' '
    return ''.join(r)

def find_final_move(board, moves):
    for i,(_i0,i1) in enumerate(moves):
        if board[i1] in 'Kk':
            return i
    else:
        return -1

score_map = {
    'R': 10 * BASE,
    'r': -10 * BASE,
    'N': 4 * BASE,
    'n': -4 * BASE,
    'B': 2 * BASE,
    'b': -2 * BASE,
    'A': 2 * BASE,
    'a': -2 * BASE,
    'K': 300 * BASE,
    'k': -300 * BASE,
    'C': 4 * BASE,
    'c': -4 * BASE,
    'P': 1 * BASE,
    'p': -1 * BASE,
    ' ': 0
}

next_chess_steps = {
    'R': next_rider_steps,
    'r': next_rider_steps,
    'N': next_horse_steps,
    'n': next_horse_steps,
    'B': next_elephant_steps,
    'b': next_elephant_steps,
    'A': next_bishop_steps,
    'a': next_bishop_steps,
    'K': next_king_steps,
    'k': next_king_steps,
    'C': next_cannon_steps,
    'c': next_cannon_steps,
    'P': next_pawn_steps,
    'p': next_pawn_steps
}


class MoveTransform:
    __m2i__ = None
    __i2m__ = None
    __rotate_indices__ = None
    @staticmethod
    def m2i():
        if __class__.__m2i__ is None:
            __class__.__m2i__ = {m:i for i,m in enumerate(__class__._compute_move_ids())}
        return __class__.__m2i__

    @staticmethod
    def i2m():
        if __class__.__i2m__ is None:
            __class__.__i2m__ = {i:m for i,m in enumerate(__class__._compute_move_ids())}
        return __class__.__i2m__

    @staticmethod
    def move_to_id(move):
        return __class__.m2i()[move]

    @staticmethod
    def id_to_move(id):
        return __class__.i2m()[id]

    @staticmethod
    def onehot(move):
        action_probs = np.zeros(__class__.action_size(), dtype=np.float32)
        action_probs[__class__.m2i()[move]] = 1
        return action_probs

    @staticmethod
    def map_probs(moves, probs):
        result = np.zeros(__class__.action_size(), dtype=np.float32)
        ids = [__class__.move_to_id(x) for x in moves]
        result[ids] = probs
        return result

    @staticmethod
    def rotate_indices():
        if __class__.__rotate_indices__ is None:
            indices = np.zeros(__class__.action_size(), dtype=np.int32)
            for id,move in __class__.i2m().items():
                i0, i1 = move
                rid = __class__.m2i()[(89-i0,89-i1)]
                indices[id] = rid
            __class__.__rotate_indices__ = indices
        return __class__.__rotate_indices__

    @staticmethod
    def action_size():
        return len(__class__.m2i())

    @staticmethod
    def _compute_move_ids():
        bishop_positions = [(2,0),(6,0),(0,2),(4,2),(8,2),(2,4),(6,4)]
        adviser_positions = [(3,0),(5,0),(4,1),(3,2),(5,2)]
        bishop_positions = __class__._make_id_and_mirror(bishop_positions)
        adviser_positions = __class__._make_id_and_mirror(adviser_positions)
        def possible_moves(i0, i1):
            x0, y0 = position_2(i0)
            x1, y1 = position_2(i1)
            if x0 == x1 and y0 == y1:
                return False
            elif abs(y1-y0) == abs(x1-x0) == 2:
                return i0 in bishop_positions and i1 in bishop_positions
            elif abs(y1-y0) == abs(x1-x0) == 1:
                return i0 in adviser_positions and i1 in adviser_positions
            elif abs((y1-y0)*(x1-x0)) == 2:
                return True
            elif (y1-y0)*(x1-x0) == 0:
                return True
            else:
                return False
        moves = set()
        for i0 in range(90):
            for i1 in range(90):
                if possible_moves(i0, i1):
                    moves.add((i0, i1))
        return sorted(moves)

    @staticmethod
    def _make_id_and_mirror(positions):
        ids = [position_1(*t) for t in positions]
        ids += [89-x for x in ids]
        return set(ids)

