# Privacy Preserving Average Consensus
# by Yilin Mo, Member, IEEE, Richard M. Murray, Fellow, IEEE
# DOI 10.1109/TAC.2016.2564339, IEEE Transactions on Automatic Control

import os
import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from pyprnt import prnt

from algorithm import Algorithm

class NoiseAlgo(Algorithm):
    def __init__(self, topology, phi, time):
        self.A = topology.A
        self.agents = topology.agents
        self.phi = phi if 0 < phi < 1 else 0.5
        self.time = time
        self.tag = "{}_φ{}".format(topology.name, phi)
        self.__init_avg = self.__average(self.agents)
        self.__noises = []
        self.__agents_history = [self.agents]
        self.__consensus = 0

    def run(self, log=False):
        # Algorithm from III. PROBLEM FORMULATION
        for k in range(self.time): # this is step 4
            self.__step1(k)
            self.__step2(k)
            self.__step3(k)
        if log:
            prnt(self.__agents_history)

    def plot(self, show=False, save=False, dirname="result"):
        if len(self.__noises) == 0 or len(self.__agents_history) == 1:
            raise AssertionError("empty data to plot")

        # reorganize
        ys = np.array(self.__agents_history).T
        # error vector
        zs = ys - np.array(self.__init_avg)
        # noise vector
        ws = np.array(self.__noises).T
        # noise sum vector
        vs = copy.deepcopy(ws)
        for vv in vs:
            for i, v in enumerate(vv):
                if i == 0:
                    continue
                vv[i] += vv[i-1]

        # PLOT
        title = "noise_{}".format(self.tag)
        size = [2, 2]
        plt.figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
        plt.suptitle(title)

        # chart 1
        plt.subplot(size[0], size[1], 1)
        plt.title('State Vector')
        x = np.arange(0, self.time+1, 1)
        for i, y in enumerate(ys):
            plt.plot(x, y, label='x{}(k)'.format(i+1))
        plt.axhline(y=self.__init_avg, color='k', linestyle='dashed')
        plt.plot(self.__consensus, self.__init_avg, marker='x', markersize=10, color="black")
        plt.xlabel('k')
        plt.ylabel('xi(k)')
        plt.xlim([0, self.time])
        #  plt.ylim([-3, 3])
        plt.legend(fontsize="small")
        # chart 2
        plt.subplot(size[0], size[1], 2)
        plt.title('Error Vector')
        for i, z in enumerate(zs):
            plt.plot(x, z, label='z{}(k)'.format(i+1))
        plt.axhline(y=0, color='k', linestyle='dashed')
        plt.plot(self.__consensus, 0, marker='x', markersize=10, color="black")
        plt.xlabel('k')
        plt.ylabel('zi(k)')
        plt.xlim([0, self.time])
        #  plt.ylim([-3, 3])
        plt.legend(fontsize="small")
        # chart 3
        plt.subplot(size[0], size[1], 3)
        plt.title('Noise Vector')
        x = np.arange(0, self.time, 1)
        for i, w in enumerate(ws):
            plt.plot(x, w, label='w{}(k)'.format(i+1))
        plt.axhline(y=0, color='k', linestyle='dashed')
        plt.plot(self.__consensus, 0, marker='x', markersize=10, color="black")
        plt.xlabel('k')
        plt.ylabel('wi(k)')
        plt.xlim([0, self.time])
        #  plt.ylim([-3, 3])
        plt.legend(fontsize="small")
        # chart 4
        plt.subplot(size[0], size[1], 4)
        plt.title('Noise Sum Vector')
        for i, w in enumerate(vs):
            plt.plot(x, w, label='v{}(k)'.format(i+1))
        plt.axhline(y=0, color='k', linestyle='dashed')
        plt.plot(self.__consensus, 0, marker='x', markersize=10, color="black")
        plt.xlabel('k')
        plt.ylabel('sum(wi(k))')
        plt.xlim([0, self.time])
        #  plt.ylim([-3, 3])
        plt.legend(fontsize="small")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save:
            filename = title
            plt.savefig(os.path.join(dirname, filename) + ".png")
        if show:
            plt.show()
        plt.clf()

    def __average(self, data):
        return sum(data) / len(data)

    def __random(self, size=1):
        mean = 0
        variance = 1 # vi(k) = 1
        stdev = math.sqrt(variance) # Var(vi(k)) = σ^2, σ = √(Var(vi(k)))
        # Gaussian / Normal Distribution
        return np.random.normal(mean, stdev, size)

    def __step1(self, k):
        noises = self.__random(len(self.agents))
        self.__noises.append(noises)

    def __step2(self, k):
        if k == 0:
            w = self.__noises[0]
        else:
            w = ((self.phi ** k) * self.__noises[k]) - ((self.phi ** (k-1)) * self.__noises[k-1])
            # decide positivity or negativity for this noise
            # to achieve asymptotic sum of 0
            agent_sums = np.array(self.__noises[:-1]).sum(axis=0)
            signs = [1 if agent_sum <= 0 else -1 for agent_sum in agent_sums]
            w = np.absolute(w) * signs
        self.__noises[k] = w
        self.agents += self.__noises[k]

    def __step3(self, k):
        reshaped_agents = np.array([self.agents]).T
        new_states = self.A * reshaped_agents
        new_states = new_states.T.tolist()[0]
        self.agents = new_states
        self.__agents_history.append(new_states)

        # check if consensus is made
        new_states = [float("{0:.2f}".format(agent)) for agent in new_states]
        agents_average = float("{0:.2f}".format(self.__init_avg))
        if all(agent-agents_average == 0 for agent in new_states):
            self.__consensus = k if self.__consensus == 0 else self.__consensus

