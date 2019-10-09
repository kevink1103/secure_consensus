# Based on NUMERICAL EXAMPLES in Section VI
# Privacy Preserving Average Consensus
# by Yilin Mo, Member, IEEE, Richard M. Murray, Fellow, IEEE
# DOI 10.1109/TAC.2016.2564339, IEEE Transactions on Automatic Control

import math
import numpy as np
import matplotlib.pyplot as plt
from pyprnt import prnt

# Given matrix A
A = (1 / 4) * np.matrix([
    [2, 1, 0, 0, 1],
    [1, 2, 1, 0, 0],
    [0, 1, 2, 0, 1],
    [0, 0, 0, 3, 1],
    [1, 0, 1, 1, 1]
])
# phi = 0.9
phi = 0.9

def average(data):
    return sum(data)/len(data)

def random(size=1):
    mean = 0
    variance = math.sqrt(1)
    return np.random.normal(mean, variance, size)

def step1(agents, noises, k):
    return random(len(agents))

def step2(agents, noises, k):
    if k == 0:
        noises[0] = noises[0]
    else:
        noises[k] = ((phi ** k) * noises[k]) - ((phi ** (k-1)) * noises[k-1])
    
    agents = agents + noises[k]
    return agents

def step3(agents, agents_history, noises, k):
    result = A * np.reshape(np.array(agents), (5, 1))
    return np.reshape(result, (1, 5)).tolist()[0]

def plot_chart(history, n, average):
    # reorganize
    ys = [[] for i in range(len(history[0]))]
    for xs in history:
        for i, x in enumerate(xs):
            ys[i].append(float(x))
    # error vector
    zs = [[] for i in range(len(history[0]))]
    for i, y in enumerate(ys):
        z = np.array(y) - np.array(average)
        zs[i] = z.tolist()

    # plot
    plt.figure(num=None, figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')
    # chart 1
    plt.subplot(1, 2, 1)
    plt.title('State Vector')
    x = np.arange(0, n, 1)
    for i, y in enumerate(ys):
        plt.plot(x, y, label='x{}(k)'.format(i+1))
    plt.axhline(y=average, color='k', linestyle='dashed')
    plt.xlabel('k')
    plt.ylabel('xi(k)')
    plt.xlim([0, n])
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
    plt.xlim([0, n])
    plt.ylim([-3, 3])
    plt.legend()

    # show
    plt.tight_layout()
    plt.show()

def main():
    # Initial States of Agents
    agents = [-1.4, -0.8, 1.2, 0.7, -0.5]
    # agents = [10, 20, -12, 9, -22]
    agents_history = []
    noises = []
    initial_avg = average(agents)
    time = 50

    for k in range(time):
        step1_result = step1(agents, noises, k)
        noises.append(step1_result)
        # prnt("1", step1_result)

        step2_result = step2(agents, noises, k)
        agents = step2_result
        # prnt("2", step2_result)

        step3_result = step3(agents, agents_history, noises, k)
        agents = step3_result
        agents_history.append(agents)
        # prnt("3", step3_result)
        # break
    # prnt(agents_history)
    # prnt(initial_avg)
    plot_chart(agents_history, time, initial_avg)

if __name__ == "__main__":
    main()
