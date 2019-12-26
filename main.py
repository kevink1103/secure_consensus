from pyprnt import prnt

from algorithm.Topology import *
from algorithm.NormalAlgo import *
from algorithm.NoiseAlgo import *
from algorithm.CryptoAlgo import *

# TOPOLOGIES = [Paper, Mesh, Ring, Star, FullyConnected, Line, Tree]
TOPOLOGIES = [Tree]
TIME = 60
# Log Option
LOG = True
# Plot Options
SHOW = True
SAVE = False

def normal(topology):
    tag = "{}".format(topology.name)

    algo = NormalAlgo(topology.A, topology.agents, TIME)
    algo.run(log=LOG)
    algo.plot(show=SHOW, save=SAVE, tag=tag)

def noise(topology):
    # phis = [float("0.{}".format(i)) for i in range(1, 10)] # 0 < φ < 1
    phis = [0.9]
    
    for phi in phis:
        tag = "{}_φ{}".format(topology.name, phi)

        algo = NoiseAlgo(topology.A, topology.agents, phi, TIME)
        algo.run(log=LOG)
        algo.plot(show=SHOW, save=SAVE, tag=tag)

def crypto():
    agents = [1, 2, 4, 8]
    algo = CryptoAlgo(agents, 50)
    algo.run(log=LOG)

def main():
    for topology in TOPOLOGIES:
        topology = topology()
        # prnt(topology)

        # normal(topology)
        noise(topology)
    # crypto()

if __name__ == "__main__":
    main()
