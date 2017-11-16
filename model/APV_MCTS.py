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
from collections import namedtuple
import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

import utils.go as go
from utils.features import extract_features,bulk_extract_features
from utils.strategies import select_weighted_random,select_most_likely

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
c_PUCT = 5
QueueItem = namedtuple("QueueItem", "feature future")

class NetworkAPI(object):

    def __init__(self, net, num_playouts=1600):
        self.net = net
        self.now_expanding = set()
        # queue size should be >= the number of semmphores
        # in order to maxout the coroutines
        # There is not rule of thumbs to choose optimal semmphores
        # And keep in mind: the more coroutines, the less (?) quality (?)
        # of the Monte Carlo Tree obtains. As my searching is less deep
        # w.r.t a sequential MCTS. However, since MCTS is a randomnized
        # algorithm that tries to approximate a value by averaging over run_many
        # random processes, the quality of the search tree is hard to define.
        # It's a trade off among time, accuracy, and the frequency of NN updates.
        self.sem = asyncio.Semaphore(64)
        self.queue = Queue(64)
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.playouts = num_playouts

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        """
        q = self.queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num> 0 or margin > 0:
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

    #@profile
    def run_many(self,bulk_features):
        #return self.net.run_many(bulk_features)
        """simulate I/O & evaluate"""
        #sleep(np.random.random()*5e-2)
        return np.random.random((len(bulk_features),362)), np.random.random((len(bulk_features),1))

class MCTSPlayerMixin(object):

    __slot__ = ["api","parent","move","prior","position","children","U",
                "N","W"]

    '''
    params:
        @ api: NetworkAPI
        @ parent: pointer to parent MCTS nodes
        @ move: the move that leads to this node
        @ prior: the probability that leads to this node
        @ position: the board diagram of this node
        @ children: the children nodes of this node
        @ U,N,W: Upper confidence,total encouters,cumulative value
    '''
    def __init__(self, network_api, parent, move, prior):
        self.api,self.parent,self.move,self.prior,self.position,\
        self.children,self.U,self.N,self.W \
        = network_api,parent,move,prior,None,{},0,0,0

    def __repr__(self):
        return f"<MCTSNode move=self.move prior=self.prior score=self.action_score is_expanded=self.is_expanded()>"

    @property
    def Q(self):
        return self.W/self.N if self.N !=0 else 0

    @property
    def action_score(self):
        return self.Q + self.U

    def virtual_loss_do(self):
        self.N += 3
        self.W -= 3

    def virtual_loss_undo(self):
        self.N -= 3
        self.W += 3

    def is_expanded(self):
        return self.position is not None

    #@profile
    def compute_position(self):
        """Evolve the game board, and return current position"""
        self.position = self.parent.position.play_move(self.move)
        return self.position

    #@profile
    def expand(self, move_probabilities, noise=True):
        """Expand leaf node"""
        if noise:
            """add dirichlet noise on the fly"""
            move_probabilities = move_probabilities*.75 + 0.25*dirichlet([0.03]*362)

        self.children = {move: MCTSPlayerMixin(self.api,self,move,prob)
            for move, prob in np.ndenumerate(np.reshape(move_probabilities[:-1],(go.N,go.N)))}
        # Pass should always be an option! Say, for example, seki.
        self.children[None] = MCTSPlayerMixin(self.api,self,None,move_probabilities[-1])

    def backup_value_single(self,value):
        """Backup value of a single tree node"""
        self.N += 1
        if self.parent is None:
            # No point in updating Q / U values for root, since they are
            # used to decide between children nodes.
            return

        # This incrementally calculates node.Q = average(Q of children),
        # given the newest Q value and the previous average of N-1 values.
        self.W, self.U = (
            self.W + value,
            c_PUCT * np.sqrt(self.parent.N) * self.prior / self.N,
        )

    def move_prob(self):
        prob = np.asarray([child.N for child in self.children.values()]) / self.N
        prob /= np.sum(prob) # ensure 1.
        return prob

    def suggest_move(self, position):
        move_prob = self.suggest_move_prob(position)
        on_board_move_prob = np.reshape(move_prob[:-1],(go.N,go.N))
        if self.position.n < 30:
            move = select_weighted_random(position, on_board_move_prob)
        else:
            move = select_most_likely(position, on_board_move_prob)
        return move

    #@profile
    def suggest_move_prob(self, position):
        """Async tree search controller"""
        start = time.time()

        if self.parent is None:
            move_probs,_ = self.api.run_many(bulk_extract_features([position]))
            self.position = position
            self.expand(move_probs[0])

        coroutine_list = []
        for _ in range(self.api.playouts):
            coroutine_list.append(self.tree_search())
        coroutine_list.append(self.api.prediction_worker())
        self.api.loop.run_until_complete(asyncio.gather(*coroutine_list))

        logger.debug(f"Searched for {(time.time() - start):.5f} seconds")
        return self.move_prob()

    async def start_tree_search(self):
        """Monte Carlo Tree search starts here!"""

        now_expanding = self.api.now_expanding
        #TODO: add proper game over condition

        while self in now_expanding:
            await asyncio.sleep(1e-4)

        if not self.is_expanded():
            """when is leaf node try evaluate and expand"""

            # add leaf node to expanding list
            now_expanding.add(self)

            # compute leaf node position on the fly
            pos = self.compute_position()

            if pos is None:
                #print("illegal move!", file=sys.stderr)
                # See go.Position.play_move for notes on detecting legality
                # In Go, illegal move means loss (or resign)

                # remove leaf node from expanding list
                now_expanding.remove(self)

                return -1*-1

            """Show thinking history for fun"""
            #logger.debug(f"Investigating following position:\n{self.position}")

            # perform dihedral manipuation
            flip_axis,num_rot = np.random.randint(2),np.random.randint(4)
            dihedral_features = extract_features(pos,dihedral=[flip_axis,num_rot])

            # push extracted dihedral features of leaf node to the evaluation queue
            future = await self.api.push_queue(dihedral_features)  # type: Future
            await future
            move_probs, value = future.result()

            # perform reversed dihedral maniputation to move_prob
            move_probs = np.append(np.reshape(np.flip(np.rot90(np.reshape(\
            move_probs[:-1],(go.N,go.N)),4-num_rot),axis=flip_axis),(go.N**2,)),move_probs[-1])

            # expand by move probabilities
            self.expand(move_probs)

            # remove leaf node from expanding list
            now_expanding.remove(self)

            # must invert, because alternative layer has opposite objective
            return value[0]*-1

        else:
            """node has already expanded. Enter select phase."""

            # add virtual loss to current node
            self.virtual_loss_do()

            # select child node with maximum action scroe
            child = max(self.children.values(), key=lambda node: node.action_score)

            # add virtual loss to child node
            child.virtual_loss_do()

            # start child's tree search
            value = await child.start_tree_search()

            child.virtual_loss_undo()

            child.backup_value_single(value*-1)

            # subtract virtual loss imposed at the beginning
            self.virtual_loss_undo()

            # back up value just for current node
            self.backup_value_single(value)

            # must invert
            return value*-1

    async def tree_search(self):
        """Independent tree search, stands for one simulation"""

        self.api.running_simulation_num += 1

        # reduce parallel search number
        with await self.api.sem:

            value = await self.start_tree_search()

            #logger.debug(f"value: {value}")
            #logger.debug(f'Current running threads : {RUNNING_SIMULATION_NUM}')

            self.api.running_simulation_num  -= 1

            return value
