from pyprnt import prnt

from algorithm.Topology import *
from algorithm.NoiseAlgo import *

def noise():
    # topologies = [BasicMesh, Ring, Star, FullyConnected, Line, Tree]
    topologies = [BasicMesh]
    # phis = [float("0.{}".format(i)) for i in range(1, 10)] # 0 < φ < 1
    phis = [0.9]
    time = 50

    for topology in topologies:
        topology = topology()
        topology_name = topology.__class__.__name__
        prnt(topology)

        for phi in phis:
            tag = "{}_φ{}".format(topology_name, phi)

            algo = NoiseAlgo(topology.A, topology.agents, phi, time) # load data into algo
            algo.run(log=False)
            algo.plot(show=True, save=False, tag=tag)

def main():
    noise()

if __name__ == "__main__":
    main()
