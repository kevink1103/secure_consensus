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
    def __init__(self, A, agents, phi, time):
        self.A = A
        self.agents = agents
        self.phi = phi
        self.time = time
        self.__init_avg = self.__average(self.agents)
        self.__noises = []
        self.__agents_history = []

    def run(self, log=False):
        # Algorithm from III. PROBLEM FORMULATION
        for k in range(self.time): # this is step 4
            self.__step1(k)
            self.__step2(k)
            self.__step3(k)
        if log:
            prnt(self.__agents_history)

    def plot(self, show=False, save=False, tag="figure"):
        if len(self.__noises) == 0 or len(self.__agents_history) == 0:
            return

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
        # noise vector
        ws = [[] for i in range(len(self.__noises[0]))] # Empty
        for ww in self.__noises:
            for i, w in enumerate(ww):
                ws[i].append(float(w))
        # noise sum vector
        vs = copy.deepcopy(ws)
        for vv in vs:
            for i, v in enumerate(vv):
                if i == 0:
                    continue
                vv[i] += vv[i-1]

        # PLOT
        size = [2, 2]
        plt.figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
        plt.suptitle(tag)
        # chart 1
        plt.subplot(size[0], size[1], 1)
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
        plt.subplot(size[0], size[1], 2)
        plt.title('Error Vector')
        for i, z in enumerate(zs):
            plt.plot(x, z, label='z{}(k)'.format(i+1))
        plt.axhline(y=0, color='k', linestyle='dashed')
        plt.xlabel('k')
        plt.ylabel('zi(k)')
        plt.xlim([0, self.time])
        plt.ylim([-3, 3])
        plt.legend()
        # chart 3
        plt.subplot(size[0], size[1], 3)
        plt.title('Noise Vector')
        for i, w in enumerate(ws):
            plt.plot(x, w, label='w{}(k)'.format(i+1))
        plt.axhline(y=0, color='k', linestyle='dashed')
        plt.xlabel('k')
        plt.ylabel('wi(k)')
        plt.xlim([0, self.time])
        plt.ylim([-3, 3])
        plt.legend()
        # chart 4
        plt.subplot(size[0], size[1], 4)
        plt.title('Noise Sum Vector')
        for i, w in enumerate(vs):
            plt.plot(x, w, label='v{}(k)'.format(i+1))
        plt.axhline(y=0, color='k', linestyle='dashed')
        plt.xlabel('k')
        plt.ylabel('vi(k)')
        plt.xlim([0, self.time])
        plt.ylim([-3, 3])
        plt.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if show:
            plt.show()
        if save:
            dirname = "result"
            filename = "noise_{}".format(tag)
            plt.savefig(os.path.join(dirname, filename) + ".png")
    
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
        length = len(self.agents)
        reshaped_agents = np.reshape(np.array(self.agents), (length, 1))
        reshaped_agents = self.A * reshaped_agents
        reshaped_agents = np.reshape(reshaped_agents, (1, length)).tolist()[0]
        self.agents = reshaped_agents
        self.__agents_history.append(reshaped_agents)
