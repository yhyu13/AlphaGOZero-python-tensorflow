import copy
import math
import random
import sys
import time

import gtp
import numpy as np

import utils.go as go
import utils.utilities as utils

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
            all_move_probs = policy.run_many(to_play)
            for i, pos in enumerate(to_play):
                move = select_weighted_random(pos, all_move_probs[i])
                pos.play_move(move, mutate=True)

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

