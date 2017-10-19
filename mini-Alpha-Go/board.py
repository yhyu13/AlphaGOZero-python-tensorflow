"""
A collection of classes and functions for playing certain types of
games.
"""
import random, Queue
from math import sqrt, log
import Tkinter
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import sample


class Game(object):
    """
    Base class for multi-player adversarial games.
    """

    def actions(self, state):
        raise Exception('Method must be overridden.')

    def result(self, state, action, player):
        raise Exception('Method must be overridden.')

    def terminal(self, state):
        raise Exception('Method must be overridden.')

    def next_player(self, player):
        raise Exception('Method must be overridden.')

    def outcome(self, state, player):
        raise Exception('Method must be overridden.')

class ConnectFour(Game):
    """
    Implementation of the game Connect Four, modeled as a tree search problem.

    The state is a tuple of tuples. The last element is the player whose turn
    it is, the rest of the elements are tuples that represent columns in the
    game board. The first element in each corresponds to the bottom slot in the
    game board. If a slot is not occupied then it simply is not present in the
    state representation.

    ( (), (), (), (), 1 ) Four empty columns, player 1's turn

    An action is just an integer representing a column in the game board
    (state). The player is taken from the state and the move is attributed to
    this player.
    """

    PLAYERS = (1, 2)
    HEIGHT = 4 #just a defualt value
    WIDTH = 4   #just a default value

    TARGET = 3 #just a default value

    VALUE_WIN = 1
    VALUE_LOSE = -1
    VALUE_DRAW = 0

    def __init__(self, players=PLAYERS, height=HEIGHT, width=WIDTH, target=TARGET):
        self.players = players
        self.height = height
        self.width = width
        self.target = target

    def _legal(self, state, action):
        if action not in xrange(len(state)):    #check for column match
            raise Exception('Invalid action: out of range')
        return len(state[action]) < self.height #check for row match (height in the board)

    def _streak(self, state, player, start, delta, length=0):
        # Check for out-of-bounds at low end b/c of wrapping
        row, column = start
        if row < 0 or column < 0:
            return False
        try:
            piece = state[column][row]
        except IndexError:
            return False
        if piece != player:
            return False
        # Current slot is owned by the player
        length += 1
        if length == self.target:  # Streak is already long enough
            return True
        # Continue searching,
        drow, dcolumn = delta
        return self._streak(
            state,
            player,
            (row + drow, column + dcolumn),
            delta,
            length
        )

    def better_pretty_state(self, state):
        M = np.zeros(shape=(self.height,self.width))
        i = self.height-1
        j=0
        for column in state:
            for k in column:
                M[i,j]=k
                i-=1
            i=self.height-1
            j+=1
        return M
    def state_in_a_row_for_first_NN(self, state):
        ar = [] #ar will hold a list of values  for the state in the form of [1,0,2,2,1,...]
        height=self.height
        count=0
        for i in xrange(0,height):
            for column in state:
                if len(column) > i:
                    ar.append(column[i])
                else:
                    ar.append(0)

        return ar
    def pretty_state(self, state, escape=False):
        output = ''
        for j in range(self.width):
            output += ' ' + str(j)
        output += ' '
        if escape:
            output += '\\n'
        else:
            output += '\n'
        i = self.height - 1
        while i >= 0:
            for column in state:
                if len(column) > i:
                    output += '|' + str(column[i])
                else:
                    output += '| '
            output += '|'
            if escape:
                output += '\\n'
            else:
                output += '\n'
            i -= 1
        return output

    def actions(self, state):
        return tuple(
            [i for i, _ in enumerate(state) if self._legal(state, i)] 
        )

    def result(self, state, action, player):
        if not self._legal(state, action):
            raise Exception('Illegal action')
        newstate = []
        for index, column in enumerate(state):
            if index == action:
                newstate.append(column + (player,))
            else:
                newstate.append(column)
        return tuple(newstate)

    def terminal(self, state):
        # All columns full means we are done
        if all([len(column) == self.height for column in state]):
            return True
        # A winner also means we are done
        if self.outcome(state, self.players[0]) != self.VALUE_DRAW:
            return True
        # Board is not full and no one has won so the game continues
        return False

    def next_player(self, player):
        if player not in self.players:
            raise Exception('Invalid player')
        index = self.players.index(player)
        if index < len(self.players) - 1:
            return self.players[index + 1]
        else:
            return self.players[0]

    def outcome(self, state, player):
        for ci, column in enumerate(state):
            for ri, marker in enumerate(column):
                if any((
                        self._streak(state, marker, (ri, ci), (1, 0)),
                        self._streak(state, marker, (ri, ci), (0, 1)),
                        self._streak(state, marker, (ri, ci), (1, 1)),
                        self._streak(state, marker, (ri, ci), (1, -1)),
                )):
                    # A winner was found
                    if marker == player:
                        return self.VALUE_WIN
                    else:
                        return self.VALUE_LOSE
        # No winner was found
        return self.VALUE_DRAW


class Node(object):

    def __init__(self, parent, action, state, player, game=None):
        if parent is None and game is None:
            raise Exception('No game provided')
        # Game
        self.game = game or parent.game
        # Structure
        self.parent = parent
        self.children = dict.fromkeys(self.game.actions(state))
        # Tree data
        self.action = action
        self.state = state
        # Search meta data
        self.player = player
        self.visits = 0
        self.value = 0.0

    def __iter__(self):    
        """
        A generator function. Does a pre-order traversal over the nodes
        in the tree without using recursion.
        """
        print "we are in __iter__!!!"
        active = Queue.Queue()
        active.put(self)
        while active.qsize() > 0:
            next = active.get()
            for _, child in next.children.items():
                if child is not None:
                    active.put(child)
            yield next

    def __len__(self):     
        """
        Returns the number of nodes in the tree. This requires a
        traversal, so it has O(n) running time.
        """
        n = 0
        print 'we are in __len__!!!'
        for node in self.traverse():  
            n += 1
        return n

    @property
    def weight(self):
        """
        The weight of the current node.
        """
        if self.visits == 0:
            return 0
        return self.value / float(self.visits)

    def search_weight(self, c):
        """
        Compute the UCT search weight function for this node. Defined as:

            w = Q(v') / N(v') + c * sqrt(2 * ln(N(v)) / N(v'))

        Where v' is the current node and v is the parent of the current node,
        and Q(x) is the total value of node x and N(x) is the number of visits
        to node x.
        """
        return self.weight + c * sqrt(2 * log(self.parent.visits) / self.visits)

    def search_weight_by_mAlphaGo(self, x,CNN_action, c):
        """
                Compute the mAlphaGo search weight function for this node. Defined as:

                    w = Q(v') / N(v') + c * sqrt(2 * ln(N(v)) / N(v')) + c*sqrt(2)*identifier of action_value

                Where v' is the current node and v is the parent of the current node,
                and Q(x) is the total value of node x and N(x) is the number of visits
                to node x.
                """

        identifier_func=0
        if (self.action==CNN_action):	#good match!
            identifier_func=1

        return self.weight + 0 * sqrt(2 * log(self.parent.visits) / self.visits)+ c*sqrt(2)*identifier_func/(1+self.visits)


    def actions(self):
        """
        The valid actions for the current node state.
        """
        return self.game.actions(self.state)

    def result(self, action):
        # type: (object) -> object
        """
        The state resulting from the given action taken on the current node
        state by the node player.
        """
        return self.game.result(self.state, action, self.player)

    def terminal(self):
        """
        Whether the current node state is terminal.
        """
        return self.game.terminal(self.state)

    def next_player(self):
        """
        Returns the next game player given the current node's player.
        """
        return self.game.next_player(self.player)

    def outcome(self, player=None):
        """
        Returns the game outcome for the given player (default is the node's
        player) for the node state.
        """
        p = player or self.player
        return self.game.outcome(self.state, p)

    def fully_expanded(self):
        """
        Whether all child nodes have been expanded (instantiated). Essentially
        this just checks to see if any of its children are set to None.
        """
        return not None in self.children.values()

    def expand(self,d_S_V):
        """
        Instantiates one of the unexpanded children (if there are any,
        otherwise raises an exception) and returns it.
        """

        try:
            action = self.children.keys()[self.children.values().index(None)]
        except ValueError:
            raise Exception('Node is already fully expanded')

        state = self.game.result(self.state, action, self.player)
        player = self.game.next_player(self.player)

        child = Node(self, action, state, player)
        self.children[action] = child
        #d_S_V is a dict for holding data for further analysis
        if self.parent is not None:
            if self.player is 1 and not self.terminal():
                d_S_V.setdefault(self.state,[])
                d_S_V[self.state][child.state]=child.value, child.action

        return child
    def expand1(self,d_S_V):
        """
        *********YOTAM: I added expand1 in order to try to run a BFS scan of the entire tree
        Instantiates one of the unexpanded children (if there are any,
        otherwise raises an exception) and returns it.
        """
        try:
            action = self.children.keys()[self.children.values().index(None)]
        except ValueError:
            raise Exception('Node is already fully expanded')


        state = self.game.result(self.state, action, self.player)
        player = self.game.next_player(self.player)

        child = Node(self, action, state, player)
        self.children[action] = child

        return child

    def best_child(self, c=1 / sqrt(2)):

        return max(self.children.values(), key=lambda x: x.search_weight(c))

    def best_child_by_mAlphaGo(self, CNN_action, c=1 / sqrt(2)):

        return max(self.children.values(), key=lambda x: x.search_weight_by_mAlphaGo(x,CNN_action,c))


    def best_action(self, c=1 / sqrt(2)):
        """
        Returns the action needed to reach the best child from the current
        node.
        """
        return self.best_child(c).action

    def best_action_by_mAlphaGo(self, CNN_action, c=1 / sqrt(2)):

        # Returns the action needed to reach the best child from the current
        # node.

        return self.best_child_by_mAlphaGo(CNN_action,c).action

    def max_child(self):
        """
        Returns the child with the highest value.
        """
        return max(self.children.values(), key=lambda x: x.weight)

    def simulation(self, player):
        """
        Simulates the game to completion, choosing moves in a uniformly random
        manner. The outcome of the simulation is returns as the state value for
        the given player.
        """
        st = self.state
        pl = self.player
        while not self.game.terminal(st):
            action = sample(self.game.actions(st), 1)[0]
            st = self.game.result(st, action, pl)
            pl = self.game.next_player(pl)
        return self.game.outcome(st, player)

def mcts_mAlphaGo(game,state,player,budget,d_S_V,CNN_action):
    """
        Implementation of the mAlphaGo's search algorithm
        """

    root = Node(None, None, state, player, game)
    # print "state of root", root.state
    dr = []
    t = 0
    t_a = []
    while budget:
        budget -= 1
        # Tree Policy
        child = root
        while not child.terminal():
            if not child.fully_expanded():
                child = child.expand(d_S_V)
                break
            else:
                child = child.best_child()
        # Default Policy
        delta = child.simulation(player)
        # print "player is: ", player

        # Backup
        while not child is None:
            child.visits += 1
            child.value += delta
            # if not child.parent is None:
            if child.player is 1 and not child.terminal():
                d_S_V.setdefault(child.state, {})

            for key in d_S_V:
                if child.state in d_S_V[key]:
                    # if child.player is 1:
                    d_S_V[key][child.state] = child.value, child.action


            child = child.parent  #  backprop

            # ***************START DRAWING MCTS CONVERGENCE PLOT****************
    #    if root.fully_expanded():
    #        #print root.best_action(c=0)
    #        if root.visits is not 0:
    #            dr.append(root.weight)
    #        t_a.append(t)
    #        t+=1
    #plt.plot(t_a,dr, 'bs')
    #plt.ylabel('root weight value for mAlphaGo')
    #plt.xlabel('iteration number')
    #plt.show()
            # **************STOP DRAWING MCTS CONVERGENCE PLOT*********************
    # print dr
    # print "budget now is:  ", budget
    # print "best action is: ", root.best_action(c=0)
    return root.best_action_by_mAlphaGo(CNN_action=CNN_action,c=1)


def mcts_uct(game, state, player, budget,d_S_V):
    """
    Implementation of the UCT variant of the MCTS algorithm.
    """

    root = Node(None, None, state, player, game)
    dr=[]
    t=0
    t_a=[]
    while budget:
        budget -= 1
        # Tree Policy
        child = root
        while not child.terminal():
            if not child.fully_expanded():
                child = child.expand(d_S_V)
                break
            else:
                child = child.best_child()
        # Default Policy
        delta = child.simulation(player)

        # Backup
        while not child is None:
            child.visits += 1
            child.value += delta
            #if not child.parent is None:
            if child.player is 1 and not child.terminal():
                d_S_V.setdefault(child.state, {})

            for key in d_S_V:
                if child.state in d_S_V[key]:
                        d_S_V[key][child.state]=child.value, child.action
                #print child.state,d_S_V[child.state]
            # end function and return to safe haven

            child = child.parent        # backprop

#***************START DRAWING MCTS CONVERGENCE PLOT****************
    #    if root.fully_expanded():
    #        #print root.best_action(c=0)
    #        if root.visits is not 0:
    #            dr.append(root.weight)
    #        t_a.append(t)
    #        t+=1
    #plt.plot(t_a,dr, 'ro')
    #plt.ylabel('root weight value for MCTS')
    #plt.xlabel('iteration number')
    #plt.show()
#**************STOP DRAWING MCTS CONVERGENCE PLOT*********************
    #print dr
    #print "budget now is:  ", budget
    #print "best action is: ", root.best_action(c=0)
    return root.best_action(c=1)

def BFS_scan_of_tree(game,state,player,d_S_V,child1=0,start=0):
    """
    trying to build the full game tree with backpropagating from each final state (tree leaf).
    """
    count=0
    if start==0:
        child = Node(None, None, state, player, game)
    else:
         child=child1

    tree=list()             #tmp var to keep all the nodes to expand in order to construct tree2save
    tree2save=list()        #tree2save will be the tree which we'll use for backprop and print
    while not child.terminal() or not len(tree)==0:
        while not child.fully_expanded() and not child.terminal():
            current=child.expand1(d_S_V)
            tree.append(current)
            tree2save.append(current)
            count+=1
        child=tree.pop()

#*****************NOW TRY TO UPDATE WITH BACKPROP*****************
#first, try to find every final play
    for son in tree2save:
        if son.terminal():
            son1=son
            #backprop
            while not son1 is None:
                son1.visits += 1
                son1.value -= son.outcome(son.player)
                son1 = son1.parent

#*****************STOP TRYING TO UPDATE WITH BACKPROP****************

	# I know, I know - bad programming ahead! prepare yourselves

    #***********create a matrix for examples with dimensions: (num_of_examples)X(size_of_array)***********
    X = np.zeros(shape=(count+1, 12))          #FIXME: 16 is the dimension of the board, write it like a human being!
    Y = np.zeros(shape=(count+1,5)) #Y AS A ONE HOT VECTOR, and this is the case of 3X4 board


    #************ADD ZERO STATE**************#
    X[0]=np.zeros(shape=(1,12))
    Y[0]=np.zeros(shape=(1,5))
    Y[0][0]=1 #value check for all childern gives the following weights vector:
    #(0.886959492157,0.879333732925, 0.879333732925, 0.886959492157)
    #***********FINISH ADDING ZERO STATE******
    num_of_line=1
    for sons in tree2save:
            X[num_of_line] =sons.game.state_in_a_row_for_first_NN(sons.state)
            if not sons.terminal():
                Y[num_of_line][sons.best_action(c=0)]= 1
            else:
                Y[num_of_line][4]= 1    #because 4 is the case of terminal state - 4X4 case

            num_of_line+=1

    return X,Y
