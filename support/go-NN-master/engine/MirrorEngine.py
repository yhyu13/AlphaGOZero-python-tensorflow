#!/usr/bin/python
import numpy as np
from Engine import *
from Board import *

class MirrorEngine(BaseEngine):
    def __init__(self):
        super(BaseEngine,self).__init__() 
        self.last_opponent_play = None

    def name(self):
        return "MirrorEngine"

    def version(self):
        return "1.0"

    def stone_played(self, x, y, color):
        super(MirrorEngine, self).stone_played(x, y, color)
        self.last_opponent_play = (x,y)

    def pick_move(self, color):
        if not self.opponent_passed and self.last_opponent_play:
            mirror_x = self.board.N - self.last_opponent_play[0] - 1
            mirror_y = self.board.N - self.last_opponent_play[1] - 1
            if self.board.play_is_legal(mirror_x, mirror_y, color):
                return (mirror_x, mirror_y)

        enemy_stones = (self.board.vertices == flipped_color[color])
        our_stones = (self.board.vertices == color)
        rot_enemy_stones = np.rot90(enemy_stones, 2)

        play_vertices = np.logical_and(rot_enemy_stones, np.logical_not(our_stones))
        play_vertices =  np.logical_and(play_vertices, np.logical_not(enemy_stones))

        for x in xrange(self.board.N):
            for y in xrange(self.board.N):
                if play_vertices[x,y] and self.board.play_is_legal(x, y, color):
                    return (x,y)

        center = (self.board.N/2, self.board.N/2)
        if self.board[center] == Color.Empty and self.board.play_is_legal(center[0], center[1], color):
            return center

        return None


if __name__ == '__main__':
    import sys
    import os
    from GTP import GTP

    # Redirect stuff that would normally go to stdout
    # and stderr to a file.
    fclient = sys.stdout
    logfile = "log_mirror.txt"
    sys.stdout = sys.stderr = open(logfile, 'w', 0) # 0 = unbuffered

    engine = MirrorEngine()
    gtp = GTP(engine, fclient)
    gtp.loop()

