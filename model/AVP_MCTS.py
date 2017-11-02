import copy
import math
import sys
import time
import numpy as np

import utils.go as go
from utils.features import extract_features,bulk_extract_features

from multiprocessing import Pool 
from numpy.random import gamma

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
c_PUCT = 5

class MCTSPlayerMixin(object):
    
    def __init__(self, policy_network, parent, move, prior):
        self.policy_network = policy_network
        self.parent = parent # pointer to another MCTSNode
        self.move = move # the move that led to this node
        self.prior = prior
        self.position = None # lazily computed upon expansion
        self.children = {} # map of moves to resulting MCTSNode
        self.Q = self.parent.Q if self.parent is not None else 0 # average of all outcomes involving this node
        self.U = prior # monte carlo exploration bonus
        self.N = 0 # number of times node was visited

    def __repr__(self):
        return "<MCTSNode move=%s prior=%s score=%s is_expanded=%s>" % (self.move, self.prior, self.action_score, self.is_expanded())

    @property
    def action_score(self):
        # Note to self: after adding value network, must calculate 
        # self.Q = weighted_average(avg(values), avg(rollouts)),
        # as opposed to avg(map(weighted_average, values, rollouts))
        return self.Q + self.U

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
            for move, prob in np.ndenumerate(move_probabilities)}
        # Pass should always be an option! Say, for example, seki.
        self.children[None] = MCTSPlayerMixin(self.policy_network,self,None, 0)

    def backup_value_single(self,value):
        self.N += 1
        if self.parent is None:
            # No point in updating Q / U values for root, since they are
            # used to decide between children nodes.
            return
        # This incrementally calculates node.Q = average(Q of children),
        # given the newest Q value and the previous average of N-1 values.
        self.Q, self.U = (
            self.Q + (value - self.Q) / self.N,
            c_PUCT * math.sqrt(self.parent.N) * self.prior / self.N,
        )

    def move_prob(self):
        prob = np.asarray([child.N for child in self.children.values()]) / self.N
        prob /= np.sum(prob) # ensure 1.
        return np.reshape(prob[:-1],(go.N,go.N)) # ignore the pass move, as is_move_reasonable(pos,move) will handle it

    def suggest_move_prob(self, position):
        start = time.time()
        if self.parent is None: # is the ture root node right after None initialization
            move_probs,_ = self.policy_network.run_many([position])
            self.position = position
            self.expand(move_probs[0])
            
        self.tree_search(iters=1)
        print("Searched for %s seconds" % (time.time() - start), file=sys.stderr)
        
        return self.move_prob()

    def start_tree_search(self):
        
        if not self.is_expanded(): # leaf node
            position = self.compute_position()
            if position is None:
                #print("illegal move!", file=sys.stderr)
                # See go.Position.play_move for notes on detecting legality
                # In Go, illegal move means loss (or resign)
                self.backup_value_single(-1)
                return -1*-1
            #print("Investigating following position:\n%s" % (position), file=sys.stderr)
            move_probs,value = self.policy_network.run_many([position])
            self.expand(move_probs[0])
            self.backup_value_single(value[0,0])
            return value[0,0]*-1
        else:
            all_action_score = map(lambda node: node.action_score, self.children.values())
            move2QU = {move:action_score for move,action_score in zip(self.children.keys(),all_action_score)}
            select_move = max(move2QU, key=move2QU.get)
            value = self.children[select_move].start_tree_search()
            self.backup_value_single(value)
            return value*-1
    
    def tree_search(self,iters=100):
        for _ in range(iters):
            value = self.start_tree_search()
            #print("value: %s" % value, file=sys.stderr)

