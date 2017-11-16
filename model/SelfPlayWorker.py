import utils.go as go
from utils.strategies import simulate_game_mcts,simulate_many_games,get_winrate,extract_moves

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

from contextlib import contextmanager
from time import time

@contextmanager
def timer(message):
    tick = time()
    yield
    tock = time()
    logger.info(f"{message}: {(tock - tick):.3f} seconds")


class SelfPlayWorker(object):

    def __init__(self,net):
        self.net = net
        self.N_games_per_train = 1#10
        self.N_games = 1#25000
        self.playouts = 20#1600
        self.position = go.Position(to_play=go.BLACK)
        self.final_position_collections = []
        self.dicard_game_threshold = 30 # number of moves that is considered to resign too early
        self.resign_threshold = -0.75
        self.resign_delta = 0.05
        self.total_resigned_games = 0
        self.total_false_resigned_games = 0
        self.false_positive_resign_ratio = 0.05
        self.no_resign_this_game = False
        self.num_games_to_evaluate = 10#400

    def reset_position(self):
        self.position = go.Position(to_play=go.BLACK)

    def check_resign_stat(self, agent_resigned=True, false_positive=False):

        if agent_resigned:
            self.total_resigned_games += 1
            logger.debug(f'Total Resigned Games: {self.total_resigned_games}')

            # for every ten resignment, force one game to not resign
            if self.total_resigned_games % 10 == 0:
                self.no_resign_this_game = True
                logger.debug(f'Ok, enough! No resignment in this game!')

            # increase false positive counts
            if false_positive:
                self.total_false_resigned_games += 1
                logger.debug(f'Total False Positive Resigned Games: {self.total_false_resigned_games}')

            # dynamically increase/decrease resign threshold
            if self.total_false_resigned_games / self.total_resigned_games > self.false_positive_resign_ratio:
                self.resign_threshold  = max(-0.95,self.resign_threshold-self.resign_delta)
                logger.debug(f'Decrease Resign Threshold to: {self.resign_threshold}')
            else:
                self.resign_threshold  = min(-0.05,self.resign_threshold+self.resign_delta)
                logger.debug(f'Increase Resign Threshold to: {self.resign_threshold}')

    def run(self, lr=0.01):

        for i in range(self.N_games):
            """self play with MCTS search"""

            with timer(f"Self-Play Simulation Game #{i+1}"):
                final_position,agent_resigned,false_positive = simulate_game_mcts(self.net,self.position,\
                playouts=self.playouts,resignThreshold=self.resign_threshold,no_resign=self.no_resign_this_game)

                logger.debug(f'Game #{i+1} Final Position:\n{final_position}')

            # Discard game that resign too early
            if final_position.n <= self.dicard_game_threshold:
                logger.debug(f'Game #{i+1} ends too early, discard!')
                continue

            # add final_position to history
            self.final_position_collections.append(final_position)

            # check resignment statistics
            self.check_resign_stat(agent_resigned,false_positive)

            if (i+1) % self.N_games_per_train == 0:
                winners_training_samples, losers_training_samples = extract_moves(self.final_position_collections)
                self.net.train(winners_training_samples, direction=1.,lrn_rate=lr)
                self.net.train(losers_training_samples, direction=-1.,lrn_rate=lr)
                self.final_position_collections = []

            # reset game board
            self.reset_position()

    def evaluate_model(self,best_model):
        self.reset_position()
        final_positions = simulate_many_games(self.net,best_model,[self.position]*self.num_games_to_evaluate)
        win_ratio = get_winrate(final_positions)
        if win_ratio < 0.55:
            logger.info(f'Previous Generation win by {win_ratio}% the game! 姜还是老得辣!')
            self.net = best_model
        else:
            logger.info(f'Current Generation win by {win_ratio}% the game! 青出于蓝而胜于蓝!')
        self.reset_position()

    def evaluate_testset(self,test_dataset):
        with timer("test set evaluation"):
            self.net.test(test_dataset,proportion=.1,force_save_model=True)
