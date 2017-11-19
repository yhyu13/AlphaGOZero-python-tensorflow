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
fixed_depth = 30
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
        self.sem = asyncio.Semaphore(16)
        self.queue = Queue(16)
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.playouts = num_playouts # the more playouts the better

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
                '''
                greedy_move = divmod(np.argmax(p),go.N)
                logger.debug(f'Greedy move: {greedy_move}')
                '''
                item.future.set_result((p, v))

    async def push_queue(self, features):
        future = self.loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)
        return future

    #@profile
    def run_many(self,bulk_features):
        return self.net.run_many(bulk_features)
        """simulate data I/O & evaluate to test lower bound speed"""
        '''
        prob = np.random.random((len(bulk_features),362))
        return prob/np.sum(prob), np.random.random((len(bulk_features),1))
        '''

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
    def __init__(self, network_api, parent, move:tuple, prior:float)->None:
        self.api,self.parent,self.move,self.prior,self.position,\
        self.children,self.U,self.N,self.W \
        = network_api,parent,move,prior,None,{},0,0,0
        super().__init__()

    def __repr__(self):
        return f"<MCTSNode move=self.move prior=self.prior score=self.action_score is_expanded=self.is_expanded()>"

    @property
    def Q(self)->float:
        return self.W/self.N if self.N !=0 else 0

    @property
    def action_score(self)->float:
        return self.Q + self.U
    
    @property   
    def node_height(self)->int:
        if self.parent is None:
            return 0
        else:
            return self.parent.node_height+1
            
    def virtual_loss_do(self)->None:
        self.N += 3
        self.W -= 3

    def virtual_loss_undo(self)->None:
        self.N -= 3
        self.W += 3

    def is_expanded(self)->bool:
        return len(self.children) != 0

    #@profile
    def compute_position(self)->go.Position:
        """Evolve the game board, and return current position"""
        self.position = self.parent.position.play_move(self.move)
        return self.position

    #@profile
    def expand(self, move_probabilities:np.ndarray, noise=False)->None:
        """Expand leaf node"""
        if noise:
            """add dirichlet noise on the fly"""
            move_probabilities = move_probabilities*.75 + 0.25*dirichlet([0.03]*362)

        self.children = {move: MCTSPlayerMixin(self.api,self,move,prob)
            for move, prob in np.ndenumerate(np.reshape(move_probabilities[:-1],(go.N,go.N)))}
        # Pass should always be an option! Say, for example, seki.
        self.children[None] = MCTSPlayerMixin(self.api,self,None,move_probabilities[-1])

    def backup_value_single(self,value:float)->None:
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

    def move_prob(self)->np.ndarray:
        prob = np.asarray([child.N for child in self.children.values()]) / self.N
        prob /= np.sum(prob) # ensure 1.
        return prob

    '''
    params:
        @ move: the play excute by the player of current round
        @ pos_to_shift: the game diagram of next round
        @ make_root: make the child node a root node
        @ discard_child: effectively start a fresh tree search at each root node
        usage: shift current node to the next round child node
    '''
    def shift_node(self,move:tuple,pos_to_shift=None,make_root=True,discard_child=True)->None:
        
        if not self.is_expanded():
            # if current root is not expanded, expand to that child node with prior prob 1.
            self.children[move] = MCTSPlayerMixin(self.api,self,move,1.)

        child = self.children[move]
        self.parent,self.move,\
        self.prior,self.position,\
        self.children,self.U,self.N,self.W = \
        None if make_root else self, child.move,\
        child.prior,child.position if pos_to_shift is None else pos_to_shift,\
        {} if discard_child else child.children,child.U,child.N,child.W

    def suggest_move(self, position:go.Position, inference=True)->tuple:

        if inference:
            """Use direct NN predition (pretty weak)"""
            move_probs,value = self.api.run_many(bulk_extract_features([position]))
            move_prob = move_probs[0]
            idx = np.argmax(move_prob)
            greedy_move = divmod(idx,go.N)
            prob = move_prob[idx]
            logger.debug(f'Greedy move is: {greedy_move} with prob {prob:.3f}')
        else:
            """Use MCTS guided by NN"""
            move_prob = self.suggest_move_prob(position)        
        
        on_board_move_prob = np.reshape(move_prob[:-1],(go.N,go.N))
        if position.n < 30:
            move = select_weighted_random(position, on_board_move_prob)
        else:
            move = select_most_likely(position, on_board_move_prob)

        player = 'B' if position.to_play==1 else 'W'
        
        if inference:
            """Use direct NN value prediction (almost always 50/50)"""
            win_rate = value[0,0]/2+0.5
        else:
            """Use MCTS guided by NN average win ratio"""
            win_rate = self.children[move].Q/2+0.5
                
        logger.info(f'Win rate for player {player} is {win_rate:.4f}')

        return move

    #@profile
    def suggest_move_prob(self, position:go.Position)->np.ndarray:
        """Async tree search controller"""
        start = time.time()

        if not self.is_expanded() and self.parent is None:
            logger.debug(f'Expadning Root Node...')

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

    async def start_tree_search(self)->float:
        """Monte Carlo Tree search starts here!"""

        now_expanding = self.api.now_expanding
        #TODO: add proper game over condition

        while self in now_expanding:
            await asyncio.sleep(1e-4)
        
        """fixed depth"""
        if not self.is_expanded() or self.node_height >= fixed_depth:
            """when is leaf node try evaluate and expand"""

            # add leaf node to expanding list
            now_expanding.add(self)

            # compute leaf node position on the fly
            pos = self.position
            if pos is None:
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

    async def tree_search(self)->float:
        """Independent tree search, stands for one simulation"""

        self.api.running_simulation_num += 1

        # reduce parallel search number
        with await self.api.sem:

            value = await self.start_tree_search()

            #logger.debug(f"value: {value}")
            #logger.debug(f'Current running threads : {RUNNING_SIMULATION_NUM}')

            self.api.running_simulation_num  -= 1

            return value
