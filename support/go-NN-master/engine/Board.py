#!/usr/bin/python

import numpy as np

class Color:
    Empty = 0
    Black = 1
    White = 2

color_names = { Color.Empty:"Empty", Color.Black:"Black", Color.White:"White", }

flipped_color = { Color.Empty: Color.Empty, Color.Black: Color.White, Color.White: Color.Black }

dxdys = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class Group:
    def __init__(self, color):
        self.vertices = set([])
        self.liberties = set([])
        self.color = color

class IllegalMoveException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class Board:
    def __init__(self, N):
        self.N = N
        self.clear()

    def clear(self):
        self.vertices = np.empty((self.N, self.N), dtype=np.int32)
        self.vertices.fill(Color.Empty)
        self.groups = {} # dictionary from (x,y) tuples to Groups
        self.all_groups = set([])
        self.simple_ko_vertex = None
        self.move_list = []
        self.color_to_play = Color.Black

    def __getitem__(self, index):
        return self.vertices[index]

    def is_on_board(self, x, y):
        return x >= 0 and x < self.N and y >= 0 and y < self.N

    def adj_vertices(self, xy):
        x,y = xy
        for dx,dy in dxdys:
            adj_x,adj_y = x+dx,y+dy
            if self.is_on_board(adj_x, adj_y):
                yield adj_x,adj_y

    def merge_groups(self, a, b):
        if len(a.vertices) < len(b.vertices):
            return self.merge_groups(b,a)
        assert a.color == b.color
        a.vertices.update(b.vertices)
        a.liberties.update(b.liberties)
        for vert in b.vertices:
            self.groups[vert] = a
        self.all_groups.remove(b)
        return a

    def remove_group(self, group, capturing_color):
        self.all_groups.remove(group)
        for xy in group.vertices:
            del self.groups[xy]
            self.vertices[xy] = Color.Empty
            for adj_xy in self.adj_vertices(xy):
                if self.vertices[adj_xy] == capturing_color:
                    self.groups[adj_xy].liberties.add(xy)

    def try_play_stone(self, x, y, color, actually_execute):
        assert color == Color.White or color == Color.Black

        if not (0 <= x < self.N and 0 <= y < self.N): return False

        # no playing on top of stones
        xy = x,y
        if self.vertices[xy] != Color.Empty: return False

        # simple ko
        if self.simple_ko_vertex and xy == self.simple_ko_vertex: return False

        group = Group(color)
        group.vertices.add(xy)

        ally_groups = set([])
        enemy_groups = set([])
        self_capture = True
        for adj_xy in self.adj_vertices(xy):
            if self.vertices[adj_xy] == Color.Empty:
                group.liberties.add(adj_xy)
                self_capture = False
            else:
                adj_group = self.groups[adj_xy]
                if self.vertices[adj_xy] == color:
                    ally_groups.add(adj_group)
                    if len(adj_group.liberties) >= 2: self_capture = False
                else:
                    enemy_groups.add(adj_group)
                    if len(adj_group.liberties) == 1: self_capture = False

        # no self-capture. Move is self-capture if:
        #   vertex has no liberties and
        #   adjacent allied groups have exactly 1 liberty and
        #   all adjacent enemy groups have at least 2 liberties
        if self_capture: return False

        if not actually_execute: return True

        # move is legal, execute it
        self.vertices[xy] = color
        self.groups[xy] = group
        self.all_groups.add(group)

        # merge with allied groups
        for ally_group in ally_groups:
            ally_group.liberties.remove(xy)
            group = self.merge_groups(ally_group, group)

        # capture enemy groups
        num_captured = 0
        for enemy_group in enemy_groups:
            if len(enemy_group.liberties) == 1:
                num_captured += len(enemy_group.vertices)
                captured_vertex = next(iter(enemy_group.vertices)) # get one element from set
                self.remove_group(enemy_group, capturing_color=color)
            else:
                enemy_group.liberties.remove(xy)

        # check if recapturing would be simple ko
        if num_captured == 1 and len(group.vertices) == 1 and len(group.liberties) == 1:
            self.simple_ko_vertex = captured_vertex
        else:
            self.simple_ko_vertex = None

        self.move_list.append(xy)
        self.color_to_play = flipped_color[color]
        return True

    def play_stone(self, x, y, color):
        if not self.try_play_stone(x, y, color, actually_execute=True):
            raise IllegalMoveException("playing a %s stone at (%d,%d) is illegal" % (color_names[color], x, y))

    def play_is_legal(self, x, y, color):
        return self.try_play_stone(x, y, color, actually_execute=False)

    # Need to tell the board when someone passes so it can clear the ko state
    def play_pass(self):
        self.simple_ko_vertex = None
        self.move_list.append(None)
        self.color_to_play = flipped_color[self.color_to_play]

    #def flip_colors(self):
    #    for x in range(self.N):
    #        for y in range(self.N):
    #            self.vertices[x,y] = flipped_color[self.vertices[x,y]]

    def show(self):
        color_strings = { 
                Color.Empty: '.',
                Color.Black: '\033[31m0\033[0m',
                Color.White: '\033[37m0\033[0m' }
        for x in range(self.N): print "=",
        print
        for y in range(self.N):
            for x in range(self.N):
                if (x,y) == self.simple_ko_vertex:
                    print 'x',
                else:
                    print color_strings[self.vertices[x,y]],
            print
        for x in range(self.N): print "=",
        print

    def show_liberty_counts(self):
        color_strings = { 
                Color.Empty: ' .',
                Color.Black: '\033[31m%2d\033[0m',
                Color.White: '\033[37m%2d\033[0m' }
        for x in range(self.N): print " =",
        print
        for y in range(self.N):
            for x in range(self.N):
                s = color_strings[self.vertices[x,y]]
                if self.vertices[x,y] != Color.Empty:
                    s = s % len(self.groups[(x,y)].liberties)
                print s,
            print
        for x in range(self.N): print " =",
        print


def show_sequence(board, moves, first_color):
    board.clear()
    color = first_color
    for x,y in moves:
        legal = board.play_stone(x, y, color)
        board.show()
        color = flipped_color[color]


def test_Board():
    board = Board(5)

    print "simplest capture:"
    show_sequence(board, [(1, 0), (0, 0), (0, 1)], Color.Black)
    print "move at (0, 0) is legal?", board.play_is_legal(0, 0, Color.White)
    board.flip_colors()

    print "bigger capture:"
    show_sequence(board, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)], Color.Black)

    print "ko:"
    show_sequence(board, [(0, 1), (3, 1), (1, 0), (2, 0), (1, 2), (2, 2), (2, 1), (1, 1)], Color.Black)
    print "move at (2, 1) is legal?", board.play_is_legal(2, 1, Color.Black)
    board.show()
    board.flip_colors()
    print "fipped board:"
    board.show()

    print "self capture:"
    show_sequence(board, [(0, 1), (1, 1), (1, 0)], Color.Black)
    print "move at (0, 0) is legal?", board.play_is_legal(0, 0, Color.White)

    print "biffer self capture:"
    show_sequence(board, [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2), (1, 3), (0, 3), (1, 4)], Color.Black)
    print "move at (0, 4) is legal?", board.play_is_legal(0, 0, Color.White)



if __name__ == "__main__":
    test_Board()



