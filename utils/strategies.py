# -*- coding: future_fstrings -*-
import copy
import math
import random
import sys
import time

import gtp
import numpy as np

import utils.go as go
import utils.utilities as utils
from utils.features import extract_features,bulk_extract_features

# Draw moves from policy net until this threshold, then play moves randomly.
# This speeds up the simulation, and it also provides a logical cutoff
# for which moves to include for reinforcement learning.
POLICY_CUTOFF_DEPTH = int(go.N * go.N * 0.75) # 270 moves for a 19x19
# However, some situations end up as "dead, but only with correct play".
# Random play can destroy the subtlety of these situations, so we'll play out
# a bunch more moves from a smart network before playing out random moves.
POLICY_FINISH_MOVES = int(go.N * go.N * 0.2) # 72 moves for a 19x19

def sorted_moves(probability_array):
    coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    coords.sort(key=lambda c: probability_array[c], reverse=True)
    return coords

def is_move_reasonable(position, move):
    # A move is reasonable if it is legal and doesn't fill in your own eyes.
    return position.is_move_legal(move) and go.is_eyeish(position.board, move) != position.to_play

def select_random(position):
    possible_moves = go.ALL_COORDS[:]
    random.shuffle(possible_moves)
    for move in possible_moves:
        if is_move_reasonable(position, move):
            return move
    return None

def select_most_likely(position, move_probabilities):
    for move in sorted_moves(move_probabilities):
        if is_move_reasonable(position, move):
            return move
    return None

def select_weighted_random(position, move_probabilities):
    selection = random.random()
    cdf = move_probabilities.cumsum()
    selected_move = utils.unflatten_coords(
        cdf.searchsorted(selection, side="right"))
    if is_move_reasonable(position, selected_move):
        return selected_move
    else:
        # inexpensive fallback in case an illegal move is chosen.
        return select_most_likely(position, move_probabilities)

def simulate_game_random(position):
    """Simulates a game to termination, using completely random moves"""
    while not (position.recent[-2].move is None and position.recent[-1].move is None):
        position.play_move(select_random(position), mutate=True)

def simulate_game(policy, position):
    """Simulates a game starting from a position, using a policy network"""
    while position.n <= POLICY_CUTOFF_DEPTH:
        move_probs = policy.run(position)
        move = select_weighted_random(position, move_probs)
        position.play_move(move, mutate=True)

    simulate_game_random(position)

    return position

def simulate_many_games(policy1, policy2, positions):
    """Simulates many games in parallel, utilizing GPU parallelization to
    run the policy network for multiple games simultaneously.

    policy1 is black; policy2 is white."""

    # Assumes that all positions are on the same move number. May not be true
    # if, say, we are exploring multiple MCTS branches in parallel
    while positions[0].n <= POLICY_CUTOFF_DEPTH + POLICY_FINISH_MOVES:
        black_to_play = [pos for pos in positions if pos.to_play == go.BLACK]
        white_to_play = [pos for pos in positions if pos.to_play == go.WHITE]

        for policy, to_play in ((policy1, black_to_play),
                                (policy2, white_to_play)):
            all_move_probs = policy.run_many(bulk_extract_features(to_play))
            for i, pos in enumerate(to_play):
                if pos.n < 30:
                    move = select_weighted_random(pos, np.reshape(all_move_probs[i],(go.N,go.N)))
                else:
                    move = select_most_likely(pos, np.reshape(all_move_probs[i],(go.N,go.N)))
                pos.play_move(move, mutate=True, move_prob=all_move_probs[i])

    for pos in positions:
        simulate_game_random(pos)

    return positions


class RandomPlayerMixin:
    def suggest_move(self, position):
        return select_random(position)

class GreedyPolicyPlayerMixin:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        super().__init__()

    def suggest_move(self, position):
        move_probabilities = self.policy_network.run(position)
        return select_most_likely(position, move_probabilities)

class RandomPolicyPlayerMixin:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        super().__init__()

    def suggest_move(self, position):
        move_probabilities = self.policy_network.run(position)
        return select_weighted_random(position, move_probabilities)


from numpy.random import dirichlet
c_PUCT = 5

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

    def virtual_loss(self,add=False):
        if add:
            self.N += self.v_loss
            self.W -= self.v_loss
        else:
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
            self.Q + (value - self.Q) / self.N,
            c_PUCT * math.sqrt(self.parent.N) * self.prior / self.N,
        )

    def move_prob(self):
        prob = np.asarray([child.N for child in self.children.values()]) / self.N
        prob /= np.sum(prob) # ensure 1.
        return prob

    def suggest_move_prob(self, position, iters=2):
        start = time.time()
        if self.parent is None: # is the ture root node right after None initialization
            move_probs,_ = self.policy_network.run_many(bulk_extract_features([position]))
            self.position = position
            self.expand(move_probs[0])
            
        self.tree_search(iters=iters)
        print(f"Searched {iters} iters for {(time.time() - start)} seconds",file=sys.stderr)
        
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
            move_probs,value = self.policy_network.run_many(bulk_extract_features([position]))
            self.expand(move_probs[0])
            self.backup_value_single(value[0,0])
            return value[0,0]*-1
        else:
            '''
            all_action_score = map(lambda node: node.action_score, self.children.values())
            move2QU = {move:action_score for move,action_score in zip(self.children.keys(),all_action_score)}
            select_move = max(move2QU, key=move2QU.get)
            value = self.children[select_move].start_tree_search()
            self.backup_value_single(value)
            '''
            all_action_score = map(lambda zipped: zipped[0].Q + zipped[0].U*(0.75+0.25*(zipped[1])/(zipped[0].prior+1e-8)),\
                                   zip(self.children.values(),dirichlet([0.03]*362)))
            move2action_score = {move:action_score for move,action_score in zip(self.children.keys(),all_action_score)}
            select_move = max(move2action_score, key=move2action_score.get)
            self.children[select_move].virtual_loss(add=True)
            value = self.children[select_move].start_tree_search()
            self.children[select_move].virtual_loss(add=False)
            self.backup_value_single(value)
            return value*-1
    
    def tree_search(self,iters=1600):
        for _ in range(iters):
            value = self.start_tree_search()
            #print("value: %s" % value, file=sys.stderr)

def simulate_game_mcts(policy, position):
    """Simulates a game starting from a position, using a policy network"""
    mc_policy = MCTSPlayerMixin(policy,None,None,0)
    while position.n <= POLICY_CUTOFF_DEPTH:
        move_prob = mc_policy.suggest_move_prob(position)
        on_board_move_prob = np.reshape(move_prob[:-1],(go.N,go.N))
        if position.n < 30:
            move = select_weighted_random(position, on_board_move_prob)
        else:
            move = select_most_likely(position, on_board_move_prob)
        position.play_move(move, mutate=True,move_prob=move_prob)
        # shift to child node
        mc_policy = mc_policy.children[move]
        # discard other children nodes
        mc_policy.parent.children = None
        
    simulate_game_random(position)

    return position
