"""
*** PROFILER RESULTS ***
suggest_move_prob (/Users/yuhang/Desktop/AlphaGOZero-python-tensorflow/model/APV_MCTS.py:198)

function called 1 times

         5591653 function calls (5583728 primitive calls) in 4.388 seconds

   Ordered by: cumulative time, internal time, call count
   List reduced from 203 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.002    0.002    4.389    4.389 APV_MCTS.py:198(suggest_move_prob)
        1    0.014    0.014    4.359    4.359 {method 'run_until_complete' of 'uvloop.loop.Loop' objects}
     4455    0.010    0.000    4.317    0.001 APV_MCTS.py:298(tree_search)
9244/2919    0.054    0.000    4.287    0.001 APV_MCTS.py:219(start_tree_search)
     1320    0.027    0.000    2.290    0.002 APV_MCTS.py:166(expand)
     1320    0.555    0.000    2.170    0.002 APV_MCTS.py:171(<dictcomp>)
     3561    0.483    0.000    1.546    0.000 {built-in method builtins.max}
  1289082    0.368    0.000    1.063    0.000 APV_MCTS.py:282(<lambda>)
   477840    1.053    0.000    1.053    0.000 APV_MCTS.py:127(__init__)
  1289082    0.407    0.000    0.695    0.000 APV_MCTS.py:143(action_score)
   477840    0.471    0.000    0.564    0.000 index_tricks.py:516(__next__)
  1289082    0.288    0.000    0.288    0.000 APV_MCTS.py:139(Q)
     1320    0.006    0.000    0.138    0.000 features.py:116(extract_features)
     1600    0.004    0.000    0.136    0.000 APV_MCTS.py:159(compute_position)
     1600    0.015    0.000    0.133    0.000 go.py:357(play_move)
     1320    0.003    0.000    0.095    0.000 features.py:117(<listcomp>)
"""
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
from numpy.random import dirichlet,gamma
from collections import namedtuple
import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

import utils.go as go
from utils.features import extract_features,bulk_extract_features

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
c_PUCT = 5

NOW_EXPANDING = set()
# queue size should be >= the number of semmphores
# in order to maxout the coroutines
# There is not rule of thumbs to choose optimal semmphores
# And keep in mind: the more coroutines, the less (?) quality (?)
# of the Monte Carlo Tree obtains. As my searching is less deep
# w.r.t a sequential MCTS. However, since MCTS is a randomnized
# algorithm that tries to approximate a value by averaging over run_many
# random processes, the quality of the search tree is hard to define.
# It's a trade off among time, accuracy, and the frequency of NN updates.
SEM = asyncio.Semaphore(64)
QUEUE = Queue(64)
LOOP = asyncio.get_event_loop()
RUNNING_SIMULATION_NUM = 0
QueueItem = namedtuple("QueueItem", "feature future")

class NetworkAPI(object):

    def __init__(self, net):
        self.net = net

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        """
        global QUEUE
        global RUNNING_SIMULATION_NUM
        q = QUEUE
        margin = 10  # avoid finishing before other searches starting.
        while RUNNING_SIMULATION_NUM> 0 or margin > 0:
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
        global QUEUE
        global LOOP
        future = LOOP.create_future()
        item = QueueItem(features, future)
        await QUEUE.put(item)
        return future

    #@profile
    def run_many(self,bulk_features):
        return self.net.run_many(bulk_features)
        """simulate I/O & evaluate"""
        #sleep(np.random.random()*5e-2)
        #return np.random.random((len(bulk_features),362)), np.random.random((len(bulk_features),1))

class MCTSPlayerMixin(object):

    __slot__ = ["api","parent","move","prior","position","children","U",
                "N","W"]

    def __init__(self, network_api, parent, move, prior):
        self.api = network_api
        self.parent = parent # pointer to another MCTSNode
        self.move = move # the move that led to this node
        self.prior = prior
        self.position = None # lazily computed upon expansion
        self.children = {} # map of moves to resulting MCTSNode
        self.U,self.N,self.W = 0,0,0

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
        position = self.parent.position.play_move(self.move)
        self.position = position
        return position

    #@profile
    def expand(self, move_probabilities, noise=True):
        """Expand leaf node"""
        if noise:
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
        #self.Q = self.W/self.N

    def move_prob(self):
        prob = np.asarray([child.N for child in self.children.values()]) / self.N
        prob /= np.sum(prob) # ensure 1.
        return prob

    @profile
    def suggest_move_prob(self, position, iters=1600):
        """Async tree search controller"""
        global LOOP

        start = time.time()

        if self.parent is None:
            move_probs,_ = self.api.run_many(bulk_extract_features([position]))
            self.position = position
            self.expand(move_probs[0])

        coroutine_list = []
        for _ in range(iters):
            coroutine_list.append(self.tree_search())
        coroutine_list.append(self.api.prediction_worker())
        LOOP.run_until_complete(asyncio.gather(*coroutine_list))

        logger.debug(f"Searched for {(time.time() - start):.5f} seconds")
        return self.move_prob()

    async def start_tree_search(self):
        global NOW_EXPANDING

        #TODO: add proper game over condition

        while self in NOW_EXPANDING:
            await asyncio.sleep(1e-4)

        if not self.is_expanded(): #  is leaf node

            # add leaf node to expanding list
            NOW_EXPANDING.add(self)

            # compute leaf node position
            pos = self.compute_position()

            if pos is None:
                #print("illegal move!", file=sys.stderr)
                # See go.Position.play_move for notes on detecting legality
                # In Go, illegal move means loss (or resign)
                # subtract virtual loss imposed at the beginnning
                self.virtual_loss_undo()
                self.backup_value_single(-1)
                NOW_EXPANDING.remove(self)
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

            # subtract virtual loss imposed at the beginnning
            self.virtual_loss_undo()

            # back up value just for current tree node
            self.backup_value_single(value[0])

            # remove leaf node from expanding list
            NOW_EXPANDING.remove(self)

            # must invert, because alternative layer has opposite objective
            return value[0]*-1

        else: # not a leaf node

            # add virtual loss
            self.virtual_loss_do()

            # select child node with maximum action scroe
            child = max(self.children.values(), key=lambda node: node.action_score)

            # add virtual loss
            child.virtual_loss_do()

            value = await child.start_tree_search()

            # subtract virtual loss imposed at the beginning
            self.virtual_loss_undo()

            # back up value just for current node
            self.backup_value_single(value)

            # must invert
            return value*-1

    async def tree_search(self):
        """Independent tree search, stands for one simulation"""
        global RUNNING_SIMULATION_NUM
        global SEM

        RUNNING_SIMULATION_NUM += 1

        # reduce parallel search number
        with await SEM:

            value = await self.start_tree_search()

            #logger.debug(f"value: {value}")
            #logger.debug(f'Current running threads : {RUNNING_SIMULATION_NUM}')

            RUNNING_SIMULATION_NUM -= 1

            return value
