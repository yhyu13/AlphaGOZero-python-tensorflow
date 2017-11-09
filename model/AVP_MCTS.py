from _asyncio import Future
import asyncio
from asyncio.queues import Queue

import copy
import sys
import time
import numpy as np
from numpy.random import dirichlet
from collections import namedtuple

import utils.go as go
from utils.features import extract_features,bulk_extract_features

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
c_PUCT = 5

NOW_EXPANDING = set()
SEM = asyncio.Semaphore(8)
LOOP = asyncio.get_event_loop()
RUNNING_SIMULATION_NUM = 0
QueueItem = namedtuple("QueueItem", "feature future")
QUEUE = Queue(8)

class NetworkAPI(object):

    def __init__(self, net):
        self.net = net

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        :return:
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
            # logger.debug(f"predicting {len(item_list)} items")
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

    def run_many(self,bulk_features):
        return self.net.run_many(bulk_features)

class MCTSPlayerMixin(object):
    
    def __init__(self, network_api, parent, move, prior):
        self.api = network_api
        self.parent = parent # pointer to another MCTSNode
        self.move = move # the move that led to this node
        self.prior = prior
        self.position = None # lazily computed upon expansion
        self.children = {} # map of moves to resulting MCTSNode
        self.U = 0 # monte carlo exploration bonus
        self.N = 0 # number of times node was visited
        self.W = 0 # all outcomes involving this node
        self.v_loss = 1000

    def __repr__(self):
        return "<MCTSNode move=%s prior=%s score=%s is_expanded=%s>" % (self.move, self.prior, self.action_score, self.is_expanded())

    @property
    def Q(self):
        return self.W / self.N if self.N != 0 else 0

    @property
    def action_score(self):
        return self.Q + self.U

    @property
    def tree_height(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.tree_height+1

    def virtual_loss_do(self):
        self.N += self.v_loss
        self.W -= self.v_loss

    def virtual_loss_undo(self):
        self.N -= self.v_loss
        self.W += self.v_loss

    def is_expanded(self):
        return self.position is not None

    def compute_position(self):
        """Evolve the game board, and return current position"""
        try:
            self.position = self.parent.position.play_move(self.move)
        except:
            self.position = None
            
        return self.position

    def expand(self, move_probabilities):
        """Expand leaf node"""
        self.children = {move: MCTSPlayerMixin(self.api,self,move, prob)
            for move, prob in np.ndenumerate(np.reshape(move_probabilities[:-1],(go.N,go.N)))}

        # Pass should always be an option! Say, for example, seki.
        self.children[None] = MCTSPlayerMixin(self.api,self,None, move_probabilities[-1])

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
            self.W +value,
            c_PUCT * np.sqrt(self.parent.N) * self.prior / self.N,
        )

    def move_prob(self):
        prob = np.asarray([child.N for child in self.children.values()]) / self.N
        prob /= np.sum(prob) # ensure 1.
        return prob

    def suggest_move_prob(self, position, iters=1600):
        """Async tree search controller"""
        global LOOP
        
        start = time.time()

        if self.parent is None: 
            move_probs,_ = self.api.run_many(bulk_extract_features([position]))
            self.position = position
            self.expand(move_probs[0])
            
        coroutine_list = []
        for it in range(iters):
            cor = self.tree_search()
            coroutine_list.append(cor)
        coroutine_list.append(self.api.prediction_worker())
        LOOP.run_until_complete(asyncio.gather(*coroutine_list))

        print(f"Searched for {(time.time() - start):.5f} seconds", file=sys.stderr)
        return self.move_prob()

    async def start_tree_search(self):
        global NOW_EXPANDING
        
        #TODO: add proper game over condition

        # add virtual loss
        self.virtual_loss_do() 
        
        while self in NOW_EXPANDING:
            await asyncio.sleep(1e-4)
        
        if not self.is_expanded(): #  is leaf node

            # add leaf node to expanding list
            NOW_EXPANDING.add(self)

            # compute leaf node position
            position = self.compute_position()

            # subtract virtual loss imposed at the beginnning
            self.virtual_loss_undo() 
            
            if position is None:
                #print("illegal move!", file=sys.stderr)
                # See go.Position.play_move for notes on detecting legality
                # In Go, illegal move means loss (or resign)
                self.backup_value_single(-1)
                
                return -1*-1
            
            """Show thinking history for fun"""
            #print(f"Investigating following position:\n{position}", file=sys.stderr)

            # perform dihedral manipuation            
            flip_axis,num_rot = np.random.randint(2),np.random.randint(4)
            dihedral_features = extract_features(position,dihedral=[flip_axis,num_rot])

            # push extracted dihedral features of leaf node to the evaluation queue
            future = await self.api.push_queue(dihedral_features)  # type: Future
            await future
            move_probs, value = future.result()
            
            # perform reversed dihedral maniputation to move_prob
            move_probs = np.append(np.reshape(np.flip(np.rot90(np.reshape(move_probs[:-1],(go.N,go.N)),4-num_rot),axis=flip_axis),(go.N**2,)),move_probs[-1])

            # expand by move probabilities
            self.expand(move_probs)

            # back up value just for current tree node
            self.backup_value_single(value[0])

            # remove leaf node from expanding list
            NOW_EXPANDING.remove(self)

            # must invert, because alternative layer has opposite objective
            return value[0]*-1
        
        else: # not a leaf node

            # perform dirichlet perturbed action score
            all_action_score = map(lambda zipped: zipped[0].Q + zipped[0].U*(0.75+0.25*(zipped[1])/(zipped[0].prior+1e-8)),\
                                   zip(self.children.values(),dirichlet([0.03]*362)))
            move2action_score = {move:action_score for move,action_score in zip(self.children.keys(),all_action_score)}

            # select the move with maximum action score
            select_move = max(move2action_score, key=move2action_score.get)

            # start async tree search from child node
            value = await self.children[select_move].start_tree_search()

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
            
            #print(f"value: {value}", file=sys.stderr)
            #print(f'Current running threads : {RUNNING_SIMULATION_NUM}')
            
            RUNNING_SIMULATION_NUM -= 1
            
            return value

