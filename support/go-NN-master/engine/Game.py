from Board import *

def str_from_vertex(x, y):
    return chr(ord('a')+x) + chr(ord('a'))+y

class Game:
    def __init__(self, N, komi, rules):
        self.moves = []
        self.N = N
        self.komi = komi
        self.rules = rules
        self.result = None

    def add_move(self, move):
        self.moves.append(move)

    def set_result(self, move):
        self.result = result

    def write_sgf(self, filename):
        assert self.result != None
        with open(filename, 'w') as f:
            f.write("(;GM[1]FF[4]")
            f.write("RU[%s]SZ[%d]KM[%0.2f]\n" % self.rules, self.N, self.komi)
            f.write("RE[%s]\n" % self.result)
            color = Color.Black
            for move in moves:
                if move.is_resign(): continue
                color_str = "B" if color == Color.Black else "W" 
                vert_str = "" if move.is_pass() else str_from_vertex(move.x, move.y)
                f.write(";%s[%s]\n" % (color_str, vert_str))
                color = flipped_color[color]
            f.write(")\n")


