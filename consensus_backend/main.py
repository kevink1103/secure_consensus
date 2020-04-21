from pyprnt import prnt

from algorithm.Topology import *
from algorithm.NormalAlgo import *
from algorithm.NoiseAlgo import *
from algorithm.CryptoAlgo import *

TOPOLOGIES = [Paper, Mesh, Ring, Star, FullyConnected, Line, Tree]
#  TOPOLOGIES = [Paper]
#  TOPOLOGIES = [Ring]
TIME = 50
#  Log Option
LOG = True
# Plot Options
SHOW = False
SAVE = True

def normal(topology):
    algo = NormalAlgo(topology, TIME)
    algo.run(log=LOG)
    algo.plot(show=SHOW, save=SAVE)

def noise(topology):
    phis = [float("0.{}".format(i)) for i in range(1, 10)] # 0 < φ < 1
    #  phis = [0.9]

    for phi in phis:
        algo = NoiseAlgo(topology, phi, TIME)
        algo.run(log=LOG)
        algo.plot(show=SHOW, save=SAVE)

def crypto(topology):
    epsilons = [float("0.{}".format(i)) for i in range(1, 10)] # 0 < ε < 1
    #  epsilons = [0.5]

    for epsilon in epsilons:
        algo = CryptoAlgo(topology, epsilon, TIME)
        algo.run(log=LOG)
        algo.plot(show=SHOW, save=SAVE)

def main():
    for topology in TOPOLOGIES:
        topology = topology()
        #  prnt(topology)

        normal(topology)
        #  noise(topology)
        #  crypto(topology)

if __name__ == "__main__":
    main()
