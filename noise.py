# Privacy Preserving Average Consensus
# by Yilin Mo, Member, IEEE, Richard M. Murray, Fellow, IEEE
# DOI 10.1109/TAC.2016.2564339, IEEE Transactions on Automatic Control

import math
import numpy as np
import matplotlib.pyplot as plt
from pyprnt import prnt

from algorithm import Algorithm

class NoiseAlgo(Algorithm):
    def __init__(self):
        # Based on VI. NUMERICAL EXAMPLES
        self.A = (1 / 4) * np.matrix([
            [2, 1, 0, 0, 1],
            [1, 2, 1, 0, 0],
            [0, 1, 2, 0, 1],
            [0, 0, 0, 3, 1],
            [1, 0, 1, 1, 1]
        ])
        self.phi = 0.9
        self.agents = [-1.4, -0.8, 1.2, 0.7, -0.5] # by myself
        self.time = 50
        self.__agents_history = []
        self.__noises = []
        self.__init_avg = self.__average(self.agents)

    def run(self):
        # Algorithm from III. PROBLEM FORMULATION
        for k in range(self.time): # this is step 4
            self.__step1(k)
            self.__step2(k)
            self.__step3(k)
        prnt(self.__agents_history, both=True)

    def plot(self):
        # reorganize
        ys = [[] for i in range(len(self.__agents_history[0]))]
        for xs in self.__agents_history:
            for i, x in enumerate(xs):
                ys[i].append(float(x))
        # error vector
        zs = [[] for i in range(len(self.__agents_history[0]))]
        for i, y in enumerate(ys):
            z = np.array(y) - np.array(self.__init_avg)
            zs[i] = z.tolist()

        # plot
        plt.figure(num=None, figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')
        # chart 1
        plt.subplot(1, 2, 1)
        plt.title('State Vector')
        x = np.arange(0, self.time, 1)
        for i, y in enumerate(ys):
            plt.plot(x, y, label='x{}(k)'.format(i+1))
        plt.axhline(y=self.__init_avg, color='k', linestyle='dashed')
        plt.xlabel('k')
        plt.ylabel('xi(k)')
        plt.xlim([0, self.time])
        plt.ylim([-3, 3])
        plt.legend()
        # chart 2
        plt.subplot(1, 2, 2)
        plt.title('Error Vector')
        for i, z in enumerate(zs):
            plt.plot(x, z, label='z{}(k)'.format(i+1))
        plt.axhline(y=0, color='k', linestyle='dashed')
        plt.xlabel('k')
        plt.ylabel('zi(k)')
        plt.xlim([0, self.time])
        plt.ylim([-3, 3])
        plt.legend()

        # show
        plt.tight_layout()
        plt.show()
    
    def __average(self, data):
        return sum(data) / len(data)

    def __random(self, size=1):
        mean = 0
        variance = math.sqrt(1)
        return np.random.normal(mean, variance, size)

    def __step1(self, k):
        noises = self.__random(len(self.agents))
        self.__noises.append(noises)

    def __step2(self, k):
        if k == 0:
            self.__noises[0] = self.__noises[0]
        else:
            self.__noises[k] = ((self.phi ** k) * self.__noises[k]) - ((self.phi ** (k-1)) * self.__noises[k-1])
        self.agents += self.__noises[k]
    
    def __step3(self, k):
        reshaped_agents = np.reshape(np.array(self.agents), (5, 1))
        reshaped_agents = self.A * reshaped_agents
        reshaped_agents = np.reshape(reshaped_agents, (1, 5)).tolist()[0]
        self.agents = reshaped_agents
        self.__agents_history.append(reshaped_agents)
