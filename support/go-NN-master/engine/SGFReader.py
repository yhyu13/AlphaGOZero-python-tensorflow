#!/usr/bin/python

from Board import *

READING_NAME = 1
READING_DATA = 2

separators = set(['(', ')', ' ', '\n', '\r', '\t', ';'])

properties_taking_lists = set(['AB', # add black stone (handicap)
                               'AW', # add white stone (handicap)
                              ])

def parse_property_name(file_data, ptr):
    while file_data[ptr] in separators: 
        ptr += 1
        if ptr >= len(file_data): return (None, ptr)
    name = ''
    while file_data[ptr] != '[':
        name += file_data[ptr]
        ptr += 1
    return (name, ptr)

def parse_property_data(file_data, ptr):
    while file_data[ptr].isspace(): 
        ptr += 1
    if file_data[ptr] != '[':
        return (None, ptr)
    ptr += 1
    data = ''
    while file_data[ptr] != ']':
        data += file_data[ptr]
        ptr += 1
    ptr += 1
    return (data, ptr)

def parse_property_data_list(file_data, ptr):
    data_list = []
    while True:
        (data, ptr) = parse_property_data(file_data, ptr)
        if data == None:
            return (data_list, ptr)
        else:
            data_list.append(data)

def parse_vertex(s):
    if len(s) == 0:
        return None # pass
    if s == "tt": # GoGoD sometimes uses this to indicate a pass
        return None # We are sacrificing >19x19 support here
    x = ord(s[0]) - ord('a')
    y = ord(s[1]) - ord('a')
    return (x,y)

class SGFParser:
    def __init__(self, filename):
        with open(filename, 'r') as f: 
            self.file_data = f.read()
        self.ptr = 0

    def __iter__(self):
        return self

    def next(self):
        (property_name, self.ptr) = parse_property_name(self.file_data, self.ptr)
        if property_name == None:
            raise StopIteration
        elif property_name in properties_taking_lists:
            (property_data, self.ptr) = parse_property_data_list(self.file_data, self.ptr)
        else:
            (property_data, self.ptr) = parse_property_data(self.file_data, self.ptr)
        return (property_name, property_data)


def test_SGFParser():
    #sgf = "../data/KGS/SGFs/KGS2001/2000-10-10-1.sgf"
    sgf = "/home/greg/coding/ML/go/NN/data/GoGoD/modern_games/2007/2007-08-21g.sgf"
    parser = SGFParser(sgf)
    for property_name, property_data in parser:
        print "\"%s\" = \"%s\"" % (property_name, property_data)


class SGFReader:
    def __init__(self, filename):
        self.filename = filename
        parser = SGFParser(filename)
        self.initial_stones = []
        self.moves = []
        self.black_rank = None
        self.white_rank = None
        self.result = None
        self.board = None
        self.komi = None
        for property_name, property_data in parser:
            if property_name == "SZ": # board size
                self.board = Board(int(property_data))
            elif property_name == "AB": # black initial stones
                for vertex_str in property_data:
                    self.initial_stones.append((parse_vertex(vertex_str), Color.Black))
            elif property_name == "AW": # white initial stones
                for vertex_str in property_data:
                    self.initial_stones.append((parse_vertex(vertex_str), Color.White))
            elif property_name == "B": # black plays
                self.moves.append((parse_vertex(property_data), Color.Black))
            elif property_name == "W": # white plays
                self.moves.append((parse_vertex(property_data), Color.White))
            elif property_name == "BR": # black rank
                self.black_rank = property_data
            elif property_name == "WR": # white rank
                self.white_rank = property_data
            elif property_name == "RE": # result
                self.result = property_data
            elif property_name == "KM": # komi
                self.komi = property_data

        if not self.board:
            self.board = Board(19) # assume 19x19 if we didn't see a size

        for (x,y), color in self.initial_stones:
            self.board.play_stone(x, y, color)

        self.moves_played = 0

    def has_more(self):
        return self.moves_played < len(self.moves)

    def peek_next_move(self):
        return self.moves[self.moves_played]

    def play_next_move(self):
        move = self.moves[self.moves_played]
        self.moves_played += 1
        vertex, color = move
        if vertex:
            x,y = vertex
            self.board.play_stone(x, y, color)
        else:
            self.board.play_pass()
        return move

    def next_play_color(self):
        if self.has_more():
            return self.moves[self.moves_played][1]
        elif self.moves:
            return flipped_color[self.moves[-1][1]]
        elif self.initial_stones:
            return flipped_color[self.initial_stones[-1][1]]
        else:
            return Color.Black


def test_SGFReader():
    #sgf = "/home/greg/coding/ML/go/NN/data/KGS/SGFs/kgs-19-2008-02-new/2008-02-09-18.sgf"
    sgf = "/home/greg/coding/ML/go/NN/data/GoGoD/sets/train/1995/1995-07-01c.sgf"
    reader = SGFReader(sgf)

    print "initial position:"
    reader.board.show()

    while reader.has_more():
        print "before move, next play color is", color_names[reader.next_play_color()]
        vertex, color = reader.play_next_move()
        print "after move", vertex, "by", color_names[color], "board is"
        reader.board.show()
        print "after move, next play color is", color_names[reader.next_play_color()]

    print "Game over."

if __name__ == "__main__":
    #test_SGFParser()
    test_SGFReader()








