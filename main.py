import numpy as np

from noise import NoiseAlgo

# Based on VI. NUMERICAL EXAMPLES
##############
# Topology A #
#      1   2 #
#      |   | #
#  4 - 5 - 3 #
##############
A = (1 / 4) * np.matrix([
    [2, 1, 0, 0, 1],
    [1, 2, 1, 0, 0],
    [0, 1, 2, 0, 1],
    [0, 0, 0, 3, 1],
    [1, 0, 1, 1, 1]
])
##############
# Topology B #
#    1 - 2   #
#    |   |   #
#    4 - 3   #
##############
B = (1 / 4) * np.matrix([
    [2, 1, 0, 1],
    [1, 2, 1, 0],
    [0, 1, 2, 1],
    [1, 0, 1, 2],
])
phis = np.arange(0.1, 1, 0.1) # 0 < φ < 1
agents = [-1.4, -0.8, 1.2, 0.7, -0.5]
time = 50
print(phis)

for phi in phis:
    tag = "φ{}".format(phi)
    algo = NoiseAlgo(B, phi, agents, time, tag)
    algo.run()
    algo.plot(show=True, save=False)
