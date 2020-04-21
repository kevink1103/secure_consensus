import os
import math

import numpy as np
import matplotlib.pyplot as plt
from pyprnt import prnt

from algorithm import Algorithm

class NormalAlgo(Algorithm):
    def __init__(self, topology, time):
        self.A = topology.A
        self.agents = topology.agents
        self.time = time
        self.tag = "{}".format(topology.name)
        self.__init_avg = self.__average(self.agents)
        self.__agents_history = [self.agents]
        self.__consensus = 0

    def run(self, log=False):
        # Algorithm from III. PROBLEM FORMULATION
        for k in range(self.time): # this is step 4
            self.__step1(k)
        if log:
            prnt(self.__agents_history)

    def plot(self, show=False, save=False):
        if len(self.__agents_history) == 0:
            raise AssertionError("empty data to plot")

        # reorganize
        ys = [[] for i in range(len(self.__agents_history[0]))] # Empty
        for xs in self.__agents_history:
            for i, x in enumerate(xs):
                ys[i].append(float(x))
        # error vector
        zs = [[] for i in range(len(self.__agents_history[0]))] # Empty
        for i, y in enumerate(ys):
            z = np.array(y) - np.array(self.__init_avg)
            zs[i] = z.tolist()

        # PLOT
        title = "normal_{}".format(self.tag)
        size = [1, 2]
        plt.figure(num=None, figsize=(10, 4), dpi=100, facecolor='w', edgecolor='k')
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
        plt.ylim([-3, 3])
        plt.legend()
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
        plt.ylim([-3, 3])
        plt.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save:
            dirname = "result"
            filename = title
            plt.savefig(os.path.join(dirname, filename) + ".png")
        if show:
            plt.show()
        plt.clf()

    def __average(self, data):
        return sum(data) / len(data)

    def __step1(self, k):
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

