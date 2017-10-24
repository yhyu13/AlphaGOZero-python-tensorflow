#!/usr/bin/python
import os
import cPickle
import random
from operator import xor
from collections import defaultdict
from Board import *
from SGFReader import SGFReader
import Symmetry

np.random.seed(72936393)
zobrist_keys = np.random.randint(-2**63, 2**63-1, size=(19,19,3))
np.random.seed()

def key_from_board(board, s=0):
    vertices = board.vertices.copy()
    Symmetry.apply_symmetry_plane(vertices, s)
    return reduce(xor, (zobrist_keys[x,y,vertices[x,y]] for x in xrange(board.N) for y in xrange(board.N)))

class MoveRecord:
    def __init__(self):
        self.wins = 0
        self.losses = 0

class PositionRecord:
    def __init__(self):
        self.moves = defaultdict(MoveRecord)

def add_move_to_book(book, board, move, win):
    best_key = None
    for s in xrange(8):
        key_s = key_from_board(board, s)
        if s == 0: 
            best_key = key_s
            best_s = s
        if key_s in book:
            best_key = key_s
            best_s = s
            break
    move_s = Symmetry.get_symmetry_vertex_tuple(move, board.N, best_s)
    move_record = book[best_key].moves[move_s]
    if win:
        move_record.wins += 1
    else:
        move_record.losses += 1


def add_game_to_book(sgf, book, max_moves, rank_allowed):
    reader = SGFReader(sgf)

    if not rank_allowed(reader.black_rank) or not rank_allowed(reader.white_rank):
        print "skipping %s because of invalid rank(s) (%s and %s)" % (sgf, reader.black_rank, reader.white_rank)
        return

    if reader.result == None:
        print "skipping %s because there's no result given" % sgf
        return
    elif "B+" in reader.result:
        winner = Color.Black
    elif "W+" in reader.result:
        winner = Color.White
    else:
        print "skipping %s because I can't figure out the winner from \"%s\"" % (sgf, reader.result)
        return

    moves_played = 0
    while moves_played < max_moves and reader.has_more():
        vertex, play_color = reader.peek_next_move()
        if vertex: # if not pass
            #board_key = key_from_board(reader.board)
            #move_record = book[board_key].moves[vertex]
            #if winner == play_color:
            #    move_record.wins += 1
            #else:
            #    move_record.losses += 1
            add_move_to_book(book, reader.board, vertex, play_color == winner)
        reader.play_next_move()
        moves_played += 1

def lookup_position(book, board):
    key = None
    for s in xrange(8):
        key_s = key_from_board(board, s)
        if key_s in book:
            key = key_s
            break
    if key == None: 
        return None

    pos_record = book[key]

    ret = PositionRecord()
    for move, move_record in pos_record.moves.iteritems():
        move_s = Symmetry.get_inverse_symmetry_vertex_tuple(move, board.N, s)
        ret.moves[move_s] = move_record
    return ret

def make_book_from_GoGoD():
    book = defaultdict(PositionRecord)
    max_moves = 20
    #rank_allowed = lambda rank: rank in ['1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d']
    rank_allowed = lambda rank: True
    
    num_games = 0
    top_dir = '/home/greg/coding/ML/go/NN/data/GoGoD/modern_games'
    for sub_dir in os.listdir(top_dir):
        for fn in os.listdir(os.path.join(top_dir, sub_dir)):
            sgf = os.path.join(top_dir, sub_dir, fn)
            #print "reading sgf %s" % sgf
            add_game_to_book(sgf, book, max_moves, rank_allowed)
            num_games += 1
            if num_games % 100 == 0:
                print "finished %d games..." % num_games
            #if num_games >= 10000:
             #   return book
    return book

def prune_book(book, min_games):
    print "prune_book: initially len(book) = %d" % len(book)
    keys = book.keys()
    num_positions_deleted = 0
    for key in keys:
        num_games = sum(record.wins + record.losses for record in book[key].moves.values())
        if num_games < min_games:
            del book[key]
            num_positions_deleted += 1
    print "prune_book deleted %d positions" % num_positions_deleted
    print "prune_book: now len(book) = %d" % len(book)

def write_GoGoD_book():
    book = make_book_from_GoGoD()
    prune_book(book, min_games=2)
    print "Writing GoGoD book with %d positions." % len(book)
    with open('/home/greg/coding/ML/go/NN/engine/GoGoDBook.bin', 'w') as f:
        cPickle.dump(book, f)

def load_GoGoD_book():
    with open('/home/greg/coding/ML/go/NN/engine/GoGoDBook.bin', 'r') as f:
        book = cPickle.load(f)
    print "Loaded GoGoD book with %d positions." % len(book)
    return book

def test_book(book):
    board = Board(19)
    play_color = Color.Black

    for i in xrange(20):
        board.show()
        pos_record = lookup_position(book, board)
        if pos_record == None:
            print "book line ends"
            break
        print "known moves:"
        best_vertex = None
        best_count = 0
        for vertex in pos_record.moves:
            move_record = pos_record.moves[vertex]
            print vertex, " - wins=", move_record.wins, "; losses=", move_record.losses
            count = move_record.wins + move_record.losses
            if count > best_count:
                best_count = count
                best_vertex = vertex
        print "best_vertex =", best_vertex
        board.play_stone(best_vertex[0], best_vertex[1], play_color)
        play_color = flipped_color[play_color]

def ensure_politeness(board, xy):
    x,y = xy
    if np.all(board.vertices == Color.Empty):        
        if x < board.N/2:
            x = board.N - x - 1
        if y < board.N/2:
            y = board.N - y - 1
        if y > x:
            x,y = y,x
    return x,y

def get_book_move(board, book):
    pos_record = lookup_position(book, board)
    if pos_record:
        print "known moves:"
        best_vertex = None
        total_count = 0
        for vertex in pos_record.moves:
            move_record = pos_record.moves[vertex]
            print vertex, " - wins=", move_record.wins, "; losses=", move_record.losses
            total_count += move_record.wins + move_record.losses
        if True: #total_count >= 10:
            min_count = total_count / 10
            popular_moves = [move for move in pos_record.moves if (pos_record.moves[move].wins + pos_record.moves[move].losses > min_count)]
            print "popular moves are", popular_moves
            book_move = random.choice(popular_moves)
            book_move = ensure_politeness(board, book_move)
            return book_move
    return None


if __name__ == '__main__':
    #test_book()

    write_GoGoD_book()
    book = load_GoGoD_book()
    test_book(book)





    

