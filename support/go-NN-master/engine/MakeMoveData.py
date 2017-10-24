#!/usr/bin/python
import numpy as np
import sys
import os
import os.path
import time
import random
from SGFReader import SGFReader
from Board import *
import Features
import NPZ

def make_move_arr(x, y):
    return np.array([x,y], dtype=np.int8)

def show_plane(array):
    assert len(array.shape) == 2
    N = array.shape[0]
    print "=" * N
    for y in xrange(N):
        for x in xrange(N):
            sys.stdout.write('1' if array[x,y]==1 else '0')
        sys.stdout.write('\n')
    print "=" * array.shape[1]

def show_all_planes(array):
    assert len(array.shape) == 3
    for i in xrange(array.shape[2]):
        print "PLANE %d:" % i
        show_plane(array[:,:,i])

def show_feature_planes_and_move(feature_planes, move):
    print "FEATURE PLANES:"
    show_all_planes(feature_planes)
    print "MOVE:"
    print move

def show_batch(all_feature_planes, all_moves):
    batch_size = all_feature_planes.shape[0]
    print "MINIBATCH OF SIZE", batch_size
    for i in xrange(batch_size):
        print "EXAMPLE", i
        show_feature_planes_and_move(all_feature_planes[i,:,:,:], all_moves[i,:])

def test_feature_planes():
    board = Board(5)
    moves = [(0,0), (1,1), (2,2), (3,3), (4,4)]
    play_color = Color.Black
    for x,y in moves:
        board.show()
        feature_planes = make_feature_planes(board, play_color)
        move_arr = make_move_arr(x, y)
        show_feature_planes_and_move(feature_planes, move_arr)
        print
        board.play_stone(x, y, play_color)
        play_color = flipped_color[play_color]

def write_game_data(sgf, writer, feature_maker, rank_allowed):
    reader = SGFReader(sgf)

    color_is_good = { Color.Black: rank_allowed(reader.black_rank),
                      Color.White: rank_allowed(reader.white_rank) }
    if (not color_is_good[Color.White]) and (not color_is_good[Color.Black]):
        print "skipping game b/c of disallowed rank. ranks are B=%s, W=%s" % (reader.black_rank, reader.white_rank)
        return
    elif not color_is_good[Color.White]:
        print "ignoring white because of rank. ranks are B=%s, W=%s" % (reader.black_rank, reader.white_rank)
    elif not color_is_good[Color.Black]:
        print "ignoring black because of rank. ranks are B=%s, W=%s" % (reader.black_rank, reader.white_rank)

    try:
        while reader.has_more():
            vertex, color = reader.peek_next_move()
            if vertex and color_is_good[color]: # if not pass, and if player is good enough
                x, y = vertex
                if reader.board.play_is_legal(x, y, color):
                    feature_planes = feature_maker(reader.board, color)
                    move_arr = make_move_arr(x, y)
                    writer.push_example((feature_planes, move_arr))
                else:
                    raise IllegalMoveException("playing a %s stone at (%d,%d) is illegal" % (color_names[color], x, y))
            reader.play_next_move()
    except IllegalMoveException, e:
        print "Aborting b/c of illegal move!"
        print "sgf causing exception is %s" % sgf
        print e
        exit(-1)

def make_move_prediction_data(sgf_list, N, Nfeat, out_dir, feature_maker, rank_allowed):
    sgf_list = list(sgf_list) # make local copy to permute
    random.shuffle(sgf_list)

    writer = NPZ.RandomizingWriter(out_dir=out_dir,
            names=['feature_planes', 'moves'],
            shapes=[(N,N,Nfeat), (2,)],
            dtypes=[np.int8, np.int8],
            Nperfile=128, buffer_len=50000)

    num_games = 0
    for sgf in sgf_list:
        print "processing %s" % sgf
        write_game_data(sgf, writer, feature_maker, rank_allowed)
        num_games += 1
        if num_games % 100 == 0: print "num_games =", num_games
    writer.drain()

def make_KGS_move_data():
    N = 19
    Nfeat = 21
    feature_maker = Features.make_feature_planes_stones_4liberties_4history_ko_4captures
    rank_allowed = lambda rank: rank in ['6d', '7d', '8d', '9d', '10d', '11d', 
                                         '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p', '10p', '11p']

    for set_name in ['train', 'val', 'test']:
        base_dir = "/home/greg/coding/ML/go/NN/data/KGS/SGFs/%s" % set_name
        sgfs = [os.path.join(base_dir, sub_dir, fn) for sub_dir in os.listdir(base_dir) for fn in os.listdir(os.path.join(base_dir, sub_dir))]
        out_dir = "/home/greg/coding/ML/go/NN/data/KGS/move_examples/6dan_stones_4lib_4hist_ko_4cap_Nf21/%s" % set_name

        make_move_prediction_data(sgfs, N, Nfeat, out_dir, feature_maker, rank_allowed)

def make_GoGoD_move_data():
    N = 19
    Nfeat = 21
    feature_maker = Features.make_feature_planes_stones_4liberties_4history_ko_4captures
    rank_allowed = lambda rank: rank in [ '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d', '11d' ]

    for set_name in ['train', 'val', 'test']:
        with open('/home/greg/coding/ML/go/NN/data/GoGoD/bad_sgfs/bad_sgfs.txt', 'r') as f:
            excluded_sgfs = [line.strip() for line in f.readlines()]
            print "excluded_sgfs =\n", excluded_sgfs
        base_dir = "/home/greg/coding/ML/go/NN/data/GoGoD/sets/%s" % set_name
        sgfs = [os.path.join(base_dir, sub_dir, fn) for sub_dir in os.listdir(base_dir) for fn in os.listdir(os.path.join(base_dir, sub_dir)) if fn not in excluded_sgfs]
        out_dir = "/home/greg/coding/ML/go/NN/data/GoGoD/move_examples/stones_4lib_4hist_ko_4cap_Nf21/%s" % set_name
        make_move_prediction_data(sgfs, N, Nfeat, out_dir, feature_maker, rank_allowed)

        

if __name__ == "__main__":
    #test_feature_planes()
    #test_minibatch_read_write()
    #test_TrainingDataWrite()
    #run_PlaneTester()
    
    #make_KGS_move_data()
    make_GoGoD_move_data()
    #make_CGOS9x9_training_data()
    
    #import cProfile
    #cProfile.run('make_KGS_training_data()', sort='cumtime')

