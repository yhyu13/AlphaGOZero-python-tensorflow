from _asyncio import Future
import copy
import math
import sys
import time
import numpy as np

import utils.go as go
from utils.features import extract_features,bulk_extract_features

from numpy.random import dirichlet
import asyncio

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
c_PUCT = 5

NOW_EXPANDING = set()
PREDICTION_QUEUE = Queue(8)
SEM = asyncio.Semaphore(8)
LOOP = asyncio.get_event_loop()
RUNNING_SIMULATION_NUM = 0

class MCTSPlayerMixin(object):
    
    def __init__(self, policy_network, parent, move, prior):
        self.policy_network = policy_network
        self.parent = parent # pointer to another MCTSNode
        self.move = move # the move that led to this node
        self.prior = prior
        self.position = None # lazily computed upon expansion
        self.children = {} # map of moves to resulting MCTSNode
        self.Q = 0 # average of all outcomes involving this node
        self.U = 0 # monte carlo exploration bonus
        self.N = 0 # number of times node was visited
        self.W = 0 # all outcomes involving this node
        self.v_loss = 1000

    def __repr__(self):
        return "<MCTSNode move=%s prior=%s score=%s is_expanded=%s>" % (self.move, self.prior, self.action_score, self.is_expanded())

    @property
    def action_score(self):
        # Note to self: after adding value network, must calculate 
        # self.Q = weighted_average(avg(values), avg(rollouts)),
        # as opposed to avg(map(weighted_average, values, rollouts))
        return self.Q + self.U

    @property
    def tree_height(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.tree_height+1

    def virtual_loss(self,add=True):
        if add: # add virtual loss to encourage explore
            self.N += self.v_loss
            self.W -= self.v_loss
        else: # not add == remove virtual loss
            self.N -= self.v_loss
            self.W += self.v_loss

    def is_expanded(self):
        return self.position is not None

    def compute_position(self):
        try:
            self.position = self.parent.position.play_move(self.move)
        except:
            self.position = None
        return self.position

    def expand(self, move_probabilities):
        self.children = {move: MCTSPlayerMixin(self.policy_network,self,move, prob)
            for move, prob in np.ndenumerate(np.reshape(move_probabilities[:-1],(go.N,go.N)))}
        # Pass should always be an option! Say, for example, seki.
        self.children[None] = MCTSPlayerMixin(self.policy_network,self,None, move_probabilities[-1])

    def backup_value_single(self,value):
        self.N += 1
        if self.parent is None:
            # No point in updating Q / U values for root, since they are
            # used to decide between children nodes.
            return
        # This incrementally calculates node.Q = average(Q of children),
        # given the newest Q value and the previous average of N-1 values.
        self.Q, self.U = (
            self.W / self.N,
            c_PUCT * math.sqrt(self.parent.N) * self.prior / self.N,
        )

    def move_prob(self):
        prob = np.asarray([child.N for child in self.children.values()]) / self.N
        prob /= np.sum(prob) # ensure 1.
        return prob

    def suggest_move_prob(self, position, iters=1600):
        start = time.time()
        if self.parent is None: # is the ture root node right after None initialization
            move_probs,_ = self.policy_network.run_many([position])
            self.position = position
            self.expand(move_probs[0])
            
        coroutine_list = []
        for it in range(iters):
            cor = self.tree_search()
            coroutine_list.append(cor)
        coroutine_list.append(prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

        print(f"Searched for {(time.time() - start):.5f} seconds", file=sys.stderr)
        
        return self.move_prob()

    async def start_tree_search(self):

        #TODO: add proper game over condition

        while self in NOW_EXPANDING:
            await asyncio.sleep(1e-4)
        
        if not self.is_expanded(): #  is leaf node
            NOW_EXPANDING.add(self)
            position = self.compute_position()
            if position is None:
                #print("illegal move!", file=sys.stderr)
                # See go.Position.play_move for notes on detecting legality
                # In Go, illegal move means loss (or resign)
                self.backup_value_single(-1)
                return -1*-1
            #print(f"Investigating following position:\n{position}", file=sys.stderr)
            
            future = await self.predict(position)  # type: Future
            await future
            move_probs, value = future.result()
            
            self.expand(move_probs[0]) #
            self.backup_value_single(value[0,0]) #
            NOW_EXPANDING.remove(self)
            return value[0,0]*-1
        else:
            # sum(dirichlet([0.03]*362)) == 1
            all_action_score = map(lambda zipped: zipped[0].Q + zipped[0].U*(0.75+0.25*(zipped[1])/(zipped[0].prior+1e-8)),\
                                   zip(self.children.values(),dirichlet([0.03]*362)))
            move2action_score = {move:action_score for move,action_score in zip(self.children.keys(),all_action_score)}
            select_move = max(move2action_score, key=move2action_score.get)
            self.children[select_move].virtual_loss(add=True)
            value = await self.children[select_move].start_tree_search()
            self.children[select_move].virtual_loss(add=False)
            self.backup_value_single(value)
            return value*-1
    
    async def tree_search(self):
        RUNNING_SIMULATION_NUM += 1
        with await SEM:  # reduce parallel search number
            value = await self.start_tree_search()
            #print(f"value: {value}", file=sys.stderr)
            RUNNING_SIMULATION_NUM -= 1
            return value

