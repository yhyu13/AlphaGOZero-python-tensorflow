import time
import utils.go as go
import utils.strategies as strategies
import main
import Network
import utils.sgf_wrapper as sgf_wrapper
import utils.load_data_sets as load_data_sets
import utils.utilities as utils

'''
This file requires model to have reinforcment learning feature, will implement in model alphago model
'''

net = Network.Network(main.args,main.hps,main.args.load_model_path)
now = time.time()

N_games = 10
positions = [go.Position(to_play=go.WHITE) for i in range(N_games)]
# neural net 1 always plays "black", and variety is accomplished by
# letting white play first half the time.
strategies.simulate_many_games(net, net, positions)
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
            sgf_wrapper.replay_position(final_position))
        winner = utils.parse_game_result(final_position.result())
        for pwc in positions_w_context:
            if pwc.position.to_play == winner:
                winning_moves.append(pwc)
            else:
                losing_moves.append(pwc)
    return (load_data_sets.DataSet.from_positions_w_context(winning_moves),
            load_data_sets.DataSet.from_positions_w_context(losing_moves))

win_percentage = get_winrate(positions)
winners, losers = extract_moves(positions)
'''
net.reinforce(winners, direction=1)
net.reinforce(losers, direction=-1)
'''

