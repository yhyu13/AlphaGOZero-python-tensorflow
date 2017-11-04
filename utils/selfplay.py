import os
import sys
_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

import time
import utils.go as go
import utils.strategies as strategies
from utils.strategies import simulate_many_games, simulate_game_mcts

import main
import Network
import utils.sgf_wrapper as sgf_wrapper
import utils.load_data_sets as load_data_sets
import utils.utilities as utils
from elo.elo import expected, elo
'''
This file requires model to have reinforcment learning feature, will implement in model alphago model
'''

net = Network.Network(main.args,main.hps,main.args.load_model_path)
now = time.time()

N_games = 1
positions = [go.Position(to_play=go.BLACK) for i in range(N_games)]

# neural net 1 always plays "black", and variety is accomplished by
# letting white play first half the time.
#simulate_many_games(net, net, positions)
#simulate_many_games_mcts(net, net, positions)
position = simulate_game_mcts(net,positions[0])
positions = [position]
print('Total Time to complete ',time.time() - now)

def get_winrate(final_positions):
    black_win = [utils.parse_game_result(pos.result()) == go.BLACK
                 for pos in final_positions]
    return sum(black_win) / len(black_win)

def extract_moves(final_positions):
    winning_moves = []
    losing_moves = []
    for final_position in final_positions:
        positions_w_context = utils.take_n(
            strategies.POLICY_CUTOFF_DEPTH,
            sgf_wrapper.replay_position(final_position,extract_move_probs=True))
        winner = utils.parse_game_result(final_position.result())
        for pwc in positions_w_context:
            if pwc.position.to_play == winner:
                winning_moves.append(pwc)
            else:
                losing_moves.append(pwc)
    return load_data_sets.DataSet.from_positions_w_context(winning_moves,extract_move_prob=True),\
           load_data_sets.DataSet.from_positions_w_context(losing_moves,extract_move_prob=True)

win_percentage = get_winrate(positions)
winners_training_set, losers_training_set = extract_moves(positions)

net.train(winners_training_set, direction=1.)
net.train(losers_training_set, direction=-1.)
