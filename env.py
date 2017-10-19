import numpy as np
import matplotlib.pyplot as plt

class Refereer():

    size2rule = {"5":None,"9":None,"19":None}

    def __init__(self,size):
        # implement go rules based on size (5x5,9x9,19x19)
        self.rule = size2rule[str(size)]
        raise NotImplementedError("Please Implement this method")

    def check_win(self,grid):
        # implement winning checker based on rule
        raise NotImplementedError("Please Implement this method")

class GO_ENV(object):

    self.refereer = Refereer(self.size)
    def __init__(self,size)
        self.size = size

        # 0 == no stone, -1 == black stone, 1 == white stone
        self.grid = np.zeros(self.size**2) 

    def check_win(self):
        return self.refereer.check_win(self.grid)
