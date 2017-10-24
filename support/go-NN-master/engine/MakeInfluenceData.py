#!/usr/bin/python
import numpy as np
import os
import os.path
import random
from Board import *
from SGFReader import SGFReader
#from MakeMoveData import show_plane
import Features
import NPZ

def find_vertices_reached_by_color(board, color):
    visited = np.zeros((board.N, board.N), dtype=np.bool_)
    reached = np.zeros((board.N, board.N), dtype=np.int8)

    for x in xrange(board.N):
        for y in xrange(board.N):
            if not visited[x,y] and board[x,y] == color:
                q = [(x,y)]
                visited[x,y] = True
                reached[x,y] = 1
                while q:
                    vert = q.pop()
                    for adj in board.adj_vertices(vert):
                        if not visited[adj] and (board[adj] == color or board[adj] == Color.Empty):
                            q.append(adj)
                            visited[adj] = True
                            reached[adj] = True
    return reached




def get_final_territory_map(sgf):
    reader = SGFReader(sgf)
    while reader.has_more():
        reader.play_next_move()

    #reader.board.show()

    reached_by_black = find_vertices_reached_by_color(reader.board, Color.Black)
    reached_by_white = find_vertices_reached_by_color(reader.board, Color.White)

    #print "reached_by_black:"
    #show_plane(reached_by_black)
    #print "reached_by_white:"
    #show_plane(reached_by_white)

    territory_map = reached_by_black - reached_by_white
    #print "territory_map:\n", territory_map
    return territory_map


def write_game_data(sgf, sgf_aftermath, writer, feature_maker, rank_allowed):
    final_map_black_positive = get_final_territory_map(sgf_aftermath)
    final_map_white_positive = (-1) * final_map_black_positive
    reader = SGFReader(sgf)

    if not rank_allowed(reader.black_rank) or not rank_allowed(reader.white_rank):
        #print "skipping game b/c of disallowed rank. ranks are %s, %s" % (reader.black_rank, reader.white_rank)
        return

    while True:
        feature_planes = feature_maker(reader.board, reader.next_play_color())
        final_map = final_map_black_positive if reader.next_play_color() == Color.Black else final_map_white_positive
        writer.push_example((feature_planes, final_map))
        if reader.has_more():
            reader.play_next_move()
        else:
            break

def make_KGS_influence_data():
    N = 19
    Nfeat = 15
    feature_maker = Features.make_feature_planes_stones_3liberties_4history_ko

    for set_name in ['train', 'val', 'test']:
        games_dir = "/home/greg/coding/ML/go/NN/data/KGS/influence/games/played_out/%s" % set_name
        aftermath_dir = "/home/greg/coding/ML/go/NN/data/KGS/influence/games/playouts"
        out_dir = "/home/greg/coding/ML/go/NN/data/KGS/influence/examples/stones_3lib_4hist_ko_Nf15/%s" % set_name

        writer = NPZ.RandomizingWriter(out_dir=out_dir,
                names=['feature_planes', 'final_maps'],
                shapes=[(N,N,Nfeat), (N,N)],
                dtypes=[np.int8, np.int8],
                Nperfile=128, buffer_len=50000)
    
        rank_allowed = lambda rank: rank in ['1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d',
                                             '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p', '10p']
    
        game_fns = os.listdir(games_dir)
        random.shuffle(game_fns)
        num_games = 0
        for fn in game_fns:
            print "making influence data from %s" % fn
            sgf = os.path.join(games_dir, fn)
            sgf_aftermath = os.path.join(aftermath_dir, 'played_out_' + fn)
            assert os.path.isfile(sgf_aftermath)
    
            write_game_data(sgf, sgf_aftermath, writer, feature_maker, rank_allowed)
            
            num_games += 1
            if num_games % 100 == 0: print "Finished %d games of %d" % (num_games, len(game_fns))
    
        writer.drain()



if __name__ == '__main__':
    #get_final_territory_map("/home/greg/coding/ML/go/NN/data/KGS/influence/games/playouts/played_out_2001-05-01-2.sgf")
    #make_KGS_influence_data()
    import cProfile
    cProfile.run('make_KGS_influence_data()', sort='cumtime')



