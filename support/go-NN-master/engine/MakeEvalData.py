#!/usr/bin/python
import numpy as np
import os
import os.path
import random
from Board import *
from SGFReader import SGFReader
import Features
import NPZ


def write_game_data(sgf, writer, feature_maker, rank_allowed, komi_allowed):
    reader = SGFReader(sgf)

    if not rank_allowed(reader.black_rank) or not rank_allowed(reader.white_rank):
        print "skipping %s b/c of disallowed rank. ranks are %s, %s" % (sgf, reader.black_rank, reader.white_rank)
        return

    if reader.komi == None:
        print "skiping %s b/c there's no komi given" % sgf
        return
    komi = float(reader.komi)
    if not komi_allowed(komi):
        print "skipping %s b/c of non-allowed komi \"%s\"" % (sgf, reader.komi)

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

    while True:
        feature_planes = feature_maker(reader.board, reader.next_play_color(), komi)
        final_score = +1 if reader.next_play_color() == winner else -1
        final_score_arr = np.array([final_score], dtype=np.int8)

        writer.push_example((feature_planes, final_score_arr))
        if reader.has_more():
            reader.play_next_move()
        else:
            break

def make_KGS_eval_data():
    N = 19
    Nfeat = 22
    feature_maker = Features.make_feature_planes_stones_4liberties_4history_ko_4captures_komi

    #for set_name in ['train', 'val', 'test']:
    print "WARNING: ONLY DOING VAL AND TEST SETS!"
    for set_name in ['val', 'test']:
        games_dir = "/home/greg/coding/ML/go/NN/data/KGS/SGFs/%s" % set_name
        out_dir = "/home/greg/coding/ML/go/NN/data/KGS/eval_examples/stones_4lib_4hist_ko_4cap_komi_Nf22/%s" % set_name

        writer = NPZ.RandomizingWriter(out_dir=out_dir,
                names=['feature_planes', 'final_scores'],
                shapes=[(N,N,Nfeat), (1,)],
                dtypes=[np.int8, np.int8],
                Nperfile=128, buffer_len=50000)
    
        rank_allowed = lambda rank: True

        komi_allowed = lambda komi: komi in [0.5, 5.5, 6.5, 7.5]
    
        sgfs = []
        for sub_dir in os.listdir(games_dir):
            for fn in os.listdir(os.path.join(games_dir, sub_dir)):
                    sgfs.append(os.path.join(games_dir, sub_dir, fn))
        random.shuffle(sgfs)

        num_games = 0
        for sgf in sgfs:
            #print "making eval data from %s" % sgf
            write_game_data(sgf, writer, feature_maker, rank_allowed, komi_allowed)
            num_games += 1
            if num_games % 100 == 0: print "Finished %d games of %d" % (num_games, len(sgfs))
    
        writer.drain()


def komi_test():
    games_dir = "/home/greg/coding/ML/go/NN/data/KGS/SGFs/train"
    sgfs = []
    for sub_dir in os.listdir(games_dir):
        for fn in os.listdir(os.path.join(games_dir, sub_dir)):
            sgfs.append(os.path.join(games_dir, sub_dir, fn))
    random.shuffle(sgfs)
    counts = {}
    num_games = 0
    for sgf in sgfs:
        reader = SGFReader(sgf)
        print "komi =", reader.komi
        if reader.komi in counts:
            counts[reader.komi] += 1
        else:
            counts[reader.komi] = 1
        num_games += 1
        if num_games % 100 == 0:
            print "counts:", counts



if __name__ == '__main__':
    make_KGS_eval_data()
    #komi_test()



