import numpy as np

class Topology():
    def __str__(self):
        description = "{}, {}".format(self.__class__.__name__, self.agents)
        return description

    @property
    def name(self):
        return self.__class__.__name__

class Paper(Topology):
    ############## Based on VI. NUMERICAL EXAMPLES
    #  Topology  #
    #     1 - 2  #
    #     |   |  #
    # 4 - 5 - 3  #
    ##############
    def __init__(self):
        self.A = (1 / 4) * np.matrix([
            [2, 1, 0, 0, 1],
            [1, 2, 1, 0, 0],
            [0, 1, 2, 0, 1],
            [0, 0, 0, 3, 1],
            [1, 0, 1, 1, 1]
        ])
        self.agents = [-1.4, -0.8, 1.2, 0.7, -0.5]
        self.lagents = [1, 2, 4, 8, 2]
        #  self.A = (1 / 9) * np.matrix([
        #      [7, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #      [1, 7, 1, 0, 0, 0, 0, 0, 0, 0],
        #      [0, 1, 7, 0, 1, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 8, 1, 0, 0, 0, 0, 0],
        #      [0, 0, 1, 1, 6, 1, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 1, 7, 1, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 1, 7, 1, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 1, 7, 1, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 1, 7, 1],
        #      [1, 0, 0, 0, 0, 0, 0, 0, 1, 7],
        #  ])
        #  self.agents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #  self.lagents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class Mesh(Topology):
    ##############
    #  Topology  #
    # 1 - 2 - 3  #
    #  \ / \ /   #
    #   5 - 4    #
    ##############
    def __init__(self):
        self.A = (1 / 4) * np.matrix([
            [2, 1, 0, 0, 1],
            [1, 0, 1, 1, 1],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1]
        ])
        self.agents = [-1.4, -0.8, 1.2, 0.7, -0.5]
        self.lagents = [1, 2, 4, 8, 2]

class Ring(Topology):
    ##############
    #  Topology  #
    #   1 - 2    #
    #  /     \   #
    # 5 - 4 - 3  #
    ##############
    def __init__(self):
        self.A = (1 / 4) * np.matrix([
            [2, 1, 0, 0, 1],
            [1, 2, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 2, 1],
            [1, 0, 0, 1, 2]
        ])
        self.agents = [-1.4, -0.8, 1.2, 0.7, -0.5]
        self.lagents = [1, 2, 4, 8, 2]

class Star(Topology):
    ##############
    #  Topology  #
    #     1      #
    #     |      #
    # 2 - 3 - 4  #
    #    / \     #
    #   5   6    #
    ##############
    def __init__(self):
        self.A = (1 / 5) * np.matrix([
            [4, 0, 1, 0, 0, 0],
            [0, 4, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 1],
            [0, 0, 1, 4, 0, 0],
            [0, 0, 1, 0, 4, 0],
            [0, 0, 1, 0, 0, 4],
        ])
        self.agents = [-1.4, -0.8, 1.2, 0.7, -1.5, 1]
        self.lagents = [1, 2, 4, 8, 2, 5]

class FullyConnected(Topology):
    ############## Assume all connected to each other
    #  Topology  #
    #   1 - 2    #
    #  /|\ /|\   #
    # 3 ----- 4  #
    #  \|/ \|/   #
    #   5 - 6    #
    ##############
    def __init__(self):
        self.A = (1 / 5) * np.matrix([
            [0, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 0],
        ])
        self.agents = [-1.4, -0.8, 1.2, 0.7, -1.5, 1]
        self.lagents = [1, 2, 4, 8, 2, 5]

class Line(Topology):
    #####################
    #     Topology      #
    # 1 - 2 - 3 - 4 - 5 #
    #####################
    def __init__(self):
        self.A = (1 / 4) * np.matrix([
            [3, 1, 0, 0, 0],
            [1, 2, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 2, 1],
            [0, 0, 0, 1, 3],
        ])
        self.agents = [-1.4, -0.8, 1.2, 0.7, -0.5]
        self.lagents = [1, 2, 4, 8, 2]

class Tree(Topology):
    ###################
    #    Topology     #
    # 1 - 2 - 3 - 4   #
    #          \      #
    #           5     #
    ###################
    def __init__(self):
        self.A = (1 / 4) * np.matrix([
            [3, 1, 0, 0, 0],
            [1, 2, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 3, 0],
            [0, 0, 1, 0, 3],
        ])
        self.agents = [-1.4, -0.8, 1.2, 0.7, -0.5]
        self.lagents = [1, 2, 4, 8, 2]

