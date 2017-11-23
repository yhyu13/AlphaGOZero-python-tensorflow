from _asyncio import Future
import asyncio
from asyncio.queues import Queue
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from profilehooks import profile
import logging

import sys
import time
import numpy as np
from numpy.random import dirichlet
from scipy.stats import skewnorm
from collections import namedtuple, defaultdict
import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

import utils.go as go
from utils.features import extract_features, bulk_extract_features
from utils.strategies import select_weighted_random, select_most_likely
from utils.utilities import flatten_coords, unflatten_coords

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
c_PUCT = 5
virtual_loss = 3
cut_off_depth = 30
QueueItem = namedtuple("QueueItem", "feature future")
CounterKey = namedtuple("CounterKey", "board to_play depth")


class MCTSPlayerMixin(object):

    """MCTS Network Player Mix in

       Data structure:
           hash_table with each item numpy matrix of size 5x362
    """

    def __init__(self, net, num_playouts=1600):
        self.net = net
        self.now_expanding = set()
        self.expanded = set()
        # queue size should be >= the number of semmphores
        # in order to maxout the coroutines
        # There is not rule of thumbs to choose optimal semmphores
        # And keep in mind: the more coroutines, the less (?) quality (?)
        # of the Monte Carlo Tree obtains. As my searching is less deep
        # w.r.t a sequential MCTS. However, since MCTS is a randomnized
        # algorithm that tries to approximate a value by averaging over run_many
        # random processes, the quality of the search tree is hard to define.
        # It's a trade off among time, accuracy, and the frequency of NN updates.
        self.sem = asyncio.Semaphore(16)
        self.queue = Queue(16)
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.playouts = num_playouts  # the more playouts the better
        self.position = None

        self.lookup = {v: k for k, v in enumerate(['W', 'U', 'N', 'Q', 'P'])}

        self.hash_table = defaultdict(lambda: np.zeros([5, go.N**2 + 1]))

        # see super in gtp warpper as 'GtpInterface'
        super().__init__()

    """MCTS main functions

       The Asynchronous Policy Value Monte Carlo Tree Search:
       @ Q
       @ suggest_move
       @ suggest_move_mcts
       @ tree_search
       @ start_tree_search
       @ prediction_worker
       @ push_queue
    """

    def Q(self, position: go.Position, move: tuple)->float:
        if self.position is not None and move is not None:
            k = self.counter_key(position)
            q = self.hash_table[k][self.lookup['Q']][flatten_coords(move)]
            return q
        else:
            return 0

    #@profile
    def suggest_move(self, position: go.Position, inference=False)->tuple:

        self.position = position
        """Compute move prob"""
        if inference:
            """Use direct NN predition (pretty weak)"""
            move_probs, value = self.run_many(bulk_extract_features([position]))
            move_prob = move_probs[0]
            idx = np.argmax(move_prob)
            greedy_move = divmod(idx, go.N)
            prob = move_prob[idx]
            logger.debug(f'Greedy move is: {greedy_move} with prob {prob:.3f}')
        else:
            """Use MCTS guided by NN"""
            move_prob = self.suggest_move_mcts(position)

        """Select move"""
        on_board_move_prob = np.reshape(move_prob[:-1], (go.N, go.N))
        # logger.debug(on_board_move_prob)
        if position.n < 30:
            move = select_weighted_random(position, on_board_move_prob)
        else:
            move = select_most_likely(position, on_board_move_prob)

        """Get win ratio"""
        player = 'B' if position.to_play == 1 else 'W'

        if inference:
            """Use direct NN value prediction (almost always 50/50)"""
            win_rate = value[0, 0] / 2 + 0.5
        else:
            """Use MCTS guided by NN average win ratio"""
            win_rate = self.Q(position, move) / 2 + 0.5
        logger.info(f'Win rate for player {player} is {win_rate:.4f}')

        return move

    #@profile
    def suggest_move_mcts(self, position: go.Position, fixed_depth=True)->np.ndarray:
        """Async tree search controller"""
        start = time.time()

        key = self.counter_key(position)

        if not self.is_expanded(key):
            logger.debug(f'Expadning Root Node...')
            move_probs, _ = self.run_many(bulk_extract_features([position]))
            self.expand_node(key, move_probs[0])

        coroutine_list = []
        for _ in range(self.playouts):
            coroutine_list.append(self.tree_search(position))
        coroutine_list.append(self.prediction_worker())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

        if fixed_depth:
            """Limit tree search depth (not size)"""
            self.prune_hash_map_by_depth(lower_bound=position.n - 1,
                                         upper_bound=position.n + cut_off_depth)
        else:
            """Barely prune the parent nodes"""
            self.prune_hash_map_by_depth(lower_bound=position.n - 1, upper_bound=10e6)

        #logger.debug(f"Searched for {(time.time() - start):.5f} seconds")
        return self.move_prob(key)

    async def tree_search(self, position: go.Position)->float:
        """Independent MCTS, stands for one simulation"""
        self.running_simulation_num += 1

        # reduce parallel search number
        with await self.sem:
            value = await self.start_tree_search(position)
            #logger.debug(f"value: {value}")
            #logger.debug(f'Current running threads : {RUNNING_SIMULATION_NUM}')
            self.running_simulation_num -= 1

            return value

    async def start_tree_search(self, position: go.Position)->float:
        """Monte Carlo Tree search Select,Expand,Evauate,Backup"""
        now_expanding = self.now_expanding
        # TODO: add proper game over condition

        key = self.counter_key(position)

        while key in now_expanding:
            await asyncio.sleep(1e-4)

        if not self.is_expanded(key):
            """is leaf node try evaluate and expand"""
            # add leaf node to expanding list
            self.now_expanding.add(key)

            """Show thinking history for fun"""
            #logger.debug(f"Investigating following position:\n{position}")

            # perform dihedral manipuation
            flip_axis, num_rot = np.random.randint(2), np.random.randint(4)
            dihedral_features = extract_features(position, dihedral=[flip_axis, num_rot])

            # push extracted dihedral features of leaf node to the evaluation queue
            future = await self.push_queue(dihedral_features)  # type: Future
            await future
            move_probs, value = future.result()

            # perform reversed dihedral maniputation to move_prob
            move_probs = np.append(np.reshape(np.flip(np.rot90(np.reshape(
                move_probs[:-1], (go.N, go.N)), 4 - num_rot), axis=flip_axis), (go.N**2,)), move_probs[-1])

            # expand by move probabilities
            self.expand_node(key, move_probs)

            # remove leaf node from expanding list
            self.now_expanding.remove(key)

            # must invert, because alternative layer has opposite objective
            return value[0] * -1

        else:
            """node has already expanded. Enter select phase."""
            # select child node with maximum action scroe
            action_t = self.select_move_by_action_score(key, noise=True)

            # add virtual loss
            self.virtual_loss_do(key, action_t)

            # evolve game board status
            child_position = self.env_action(position, action_t)

            if child_position is not None:
                value = await self.start_tree_search(child_position)  # next move
            else:
                # None position means illegal move
                value = -1

            self.virtual_loss_undo(key, action_t)
            # on returning search path
            # update: N, W, Q, U
            self.back_up_value(key, action_t, value)

            # must invert
            if child_position is not None:
                return value * -1
            else:
                # illegal move doesn't mean much for the opponent
                return 0

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        """
        q = self.queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(1e-3)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            #logger.debug(f"predicting {len(item_list)} items")
            bulk_features = np.asarray([item.feature for item in item_list])
            policy_ary, value_ary = self.run_many(bulk_features)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def push_queue(self, features):
        future = self.loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)
        return future

    """MCTS helper functioins

       @ counter_key
       @ prune_hash_map_by_depth
       @ env_action
       @ is_expanded
       @ expand_node
       @ move_prob
       @ virtual_loss_do
       @ virtual_loss_undo
       @ back_up_value
       @ run_many
       @ select_move_by_action_score
    """
    @staticmethod
    def counter_key(position: go.Position)->namedtuple:
        if position is None:
            logger.warning("Can't compress None position into a key!!!")
            raise ValueError
        return CounterKey(tuple(np.ndarray.flatten(position.board)), position.to_play, position.n)

    def prune_hash_map_by_depth(self, lower_bound=0, upper_bound=5)->None:
        targets = [key for key in self.hash_table if key.depth <
                   lower_bound or key.depth > upper_bound]
        for t in targets:
            self.expanded.discard(t)
            self.hash_table.pop(t, None)
        logger.debug(f'Prune tree nodes smaller than {lower_bound}')

    def env_action(self, position: go.Position, action_t: int)->go.Position:
        """Evolve the game board, and return current position"""
        move = unflatten_coords(action_t)
        return position.play_move(move)

    def is_expanded(self, key: namedtuple)->bool:
        """Check expanded status"""
        # logger.debug(key)
        return key in self.expanded

    #@profile
    def expand_node(self, key: namedtuple, move_probabilities: np.ndarray)->None:
        """Expand leaf node"""
        self.hash_table[key][self.lookup['P']] = move_probabilities
        self.expanded.add(key)

    def move_prob(self, key, position=None)->np.ndarray:
        """Get move prob"""
        if position is not None:
            key = self.counter_key(position)
        prob = self.hash_table[key][self.lookup['N']]
        prob /= np.sum(prob)
        return prob

    def virtual_loss_do(self, key: namedtuple, action_t: int)->None:
        self.hash_table[key][self.lookup['N']][action_t] += virtual_loss
        self.hash_table[key][self.lookup['W']][action_t] -= virtual_loss

    def virtual_loss_undo(self, key: namedtuple, action_t: int)->None:
        self.hash_table[key][self.lookup['N']][action_t] -= virtual_loss
        self.hash_table[key][self.lookup['W']][action_t] += virtual_loss

    def back_up_value(self, key: namedtuple, action_t: int, value: float)->None:
        n = self.hash_table[key][self.lookup['N']][action_t] = \
            self.hash_table[key][self.lookup['N']][action_t] + 1

        w = self.hash_table[key][self.lookup['W']][action_t] = \
            self.hash_table[key][self.lookup['W']][action_t] + value

        self.hash_table[key][self.lookup['Q']][action_t] = w / n

        p = self.hash_table[key][self.lookup['P']][action_t]
        self.hash_table[key][self.lookup['U']][action_t] = c_PUCT * p * \
            np.sqrt(np.sum(self.hash_table[key][self.lookup['N']])) / (1 + n)

    #@profile
    def run_many(self, bulk_features):
        return self.net.run_many(bulk_features)
        """simulate data I/O & evaluate to test lower bound speed"""
        # Test random sample: should see expansion among all moves
        #prob = np.random.random(size=(len(bulk_features),362))
        # Test skewed sample: should see high prob for (0,0)
        #prob = np.asarray([[1]+[0]*361]*len(bulk_features))
        # return prob/np.sum(prob,axis=0), np.random.random((len(bulk_features),1))

    def select_move_by_action_score(self, key: namedtuple, noise=True)->int:

        params = self.hash_table[key]

        P = params[self.lookup['P']]
        N = params[self.lookup['N']]
        Q = params[self.lookup['W']] / (N + 1e-8)
        U = c_PUCT * P * np.sqrt(np.sum(N)) / (1 + N)

        if noise:
            action_score = Q + U * (0.75 * P + 0.25 * dirichlet([.03] * (go.N**2 + 1))) / (P + 1e-8)
        else:
            action_score = Q + U

        action_t = int(np.argmax(action_score[:-1]))
        return action_t
