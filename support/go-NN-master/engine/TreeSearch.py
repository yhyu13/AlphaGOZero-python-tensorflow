#!/usr/bin/python
import copy
from Engine import BaseEngine
from GTP import Move

# want policy network to influence evaluation????
# could modify score by policy probability, possibly in a depth-dependent way

def get_board_after_move(board, move):
    ret = copy.deepcopy(board)
    ret.play_stone(move[0], move[1], board.color_to_play)
    return ret

def minimax_eval(board, policy, value, depth):
    if depth == 0:
        score = value.evaluate(board)
        print "  "*(3-depth), "leaf node, score =", score
        return score

    moves = policy.suggest_moves(board)
    assert len(moves) > 0
    best_score = -99
    for move in moves:
        next_board = get_board_after_move(board, move)
        print "  "*(3-depth), "trying move", move
        score = -1 * minimax_eval(next_board, policy, value, depth-1)
        print "  "*(3-depth), "move", move, "has score", score
        if score > best_score: 
            best_score = score
    return best_score

def choose_move_minimax(board, policy, value, depth):
    assert depth > 0

    moves = policy.suggest_moves(board)
    best_score = -99
    best_move = None
    for move in moves:
        next_board = get_board_after_move(board, move)
        print "minimax root node: trying (%d,%d)..." % (move[0], move[1])
        score = -1 * minimax_eval(next_board, policy, value, depth-1)
        print "minimax root node: (%d,%d) gives score %f" % (move[0], move[1], score)
        if score > best_score: 
            best_score, best_move = score, move
    return best_move


# Return value of position if it's between lower and upper.
# If it's <= lower, return lower; if it's >= upper return upper.
def alphabeta_eval(board, policy, value, lower, upper, depth):
    if depth == 0:
        score = value.evaluate(board)
        print "  "*(3-depth), "leaf node, score =", score
        return score

    moves = policy.suggest_moves(board)
    assert len(moves) > 0
    for move in moves:
        next_board = get_board_after_move(board, move)
        print "  "*(3-depth), "trying move", move
        score = -1 * alphabeta_eval(next_board, policy, value, -upper, -lower, depth-1)
        print "  "*(3-depth), "move", move, "has score", score
        if score >= upper: 
            print "  "*(3-depth), "fail high!"
            return upper
        if score > lower:
            lower = score
    return lower

def choose_move_alphabeta(board, policy, value, depth):
    assert depth > 0

    moves = policy.suggest_moves(board)
    lower = -1
    upper = +1
    best_move = None
    for move in moves:
        next_board = get_board_after_move(board, move)
        print "alpha-beta root node: trying (%d,%d)..." % (move[0], move[1])
        score = -1 * alphabeta_eval(next_board, policy, value, -upper, -lower, depth-1)
        print "alpha-beta root node: (%d,%d) gives score %f" % (move[0], move[1], score)
        if score > lower:
            lower, best_move = score, move
    return best_move



class TreeSearchEngine(BaseEngine):
    def __init__(self, policy, value):
        self.policy = policy
        self.value = value
    def name(self):
        return "TreeSearch"
    def version(self):
        return "1.0"
    def pick_move(self, color):
        x,y = choose_move_alphabeta(self.board, self.policy, self.value, depth=3)
        return Move(x,y)
    def get_position_eval(self):
        return self.value.evaluate(self.board)

if __name__ == '__main__':
    import GTP
    fclient = GTP.redirect_all_output("log_engine.txt")

    import Policy
    import MoveModels
    import Eval
    import EvalModels

    #policy = Policy.AllPolicy()
    policy = Policy.TFPolicy(model=MoveModels.Conv12PosDepELU(N=19, Nfeat=21), threshold_prob=0.8, softmax_temp=1.0)
    value = Eval.TFEval(EvalModels.Conv11PosDepFC1ELU(N=19, Nfeat=21))

    engine = TreeSearchEngine(policy, value)
    
    gtp = GTP.GTP(engine, fclient)
    gtp.loop()



