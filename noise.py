import math
import numpy as np
import matplotlib.pyplot as plt
from pyprnt import prnt

from algorithm import Algorithm

class NoiseAlgo(Algorithm):
    def __init__(self):
        self.A = (1 / 4) * np.matrix([
                    [2, 1, 0, 0, 1],
                    [1, 2, 1, 0, 0],
                    [0, 1, 2, 0, 1],
                    [0, 0, 0, 3, 1],
                    [1, 0, 1, 1, 1]
                ])
        self.phi = 0.9
        self.agents = [-1.4, -0.8, 1.2, 0.7, -0.5]
        self.agents_history = []
        self.noises = []
        self.init_avg = self.__average(self.agents)
        self.time = 50

    def run(self):
        for k in range(self.time):
            self.__step1(k)
            self.__step2(k)
            self.__step3(k)
        prnt(self.agents_history)

    def plot(self):
        # reorganize
        ys = [[] for i in range(len(self.agents_history[0]))]
        for xs in self.agents_history:
            for i, x in enumerate(xs):
                ys[i].append(float(x))
        # error vector
        zs = [[] for i in range(len(self.agents_history[0]))]
        for i, y in enumerate(ys):
            z = np.array(y) - np.array(self.init_avg)
            zs[i] = z.tolist()

        # plot
        plt.figure(num=None, figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')
        # chart 1
        plt.subplot(1, 2, 1)
        plt.title('State Vector')
        x = np.arange(0, self.time, 1)
        for i, y in enumerate(ys):
            plt.plot(x, y, label='x{}(k)'.format(i+1))
        plt.axhline(y=self.init_avg, color='k', linestyle='dashed')
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
        self.noises.append(noises)

    def __step2(self, k):
        if k == 0:
            self.noises[0] = self.noises[0]
        else:
            self.noises[k] = ((self.phi ** k) * self.noises[k]) - ((self.phi ** (k-1)) * self.noises[k-1])

        self.agents += self.noises[k]
    
    def __step3(self, k):
        reshaped_agents = np.reshape(np.array(self.agents), (5, 1))
        reshaped_agents = self.A * reshaped_agents
        reshaped_agents = np.reshape(reshaped_agents, (1, 5)).tolist()[0]
        self.agents = reshaped_agents
        self.agents_history.append(reshaped_agents)
    
