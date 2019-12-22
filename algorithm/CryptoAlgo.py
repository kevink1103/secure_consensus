import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pyprnt import prnt

try:
    # For external
    from algorithm import Algorithm
    from algorithm.Prime import generate_prime_number
except:
    # For internal
    from Algorithm import Algorithm
    from Prime import generate_prime_number

class SimplePaillier:
    def __init__(self):
        keys = self.__generate_keys()
        self.public_key = keys[0]
        self.private_key = keys[1]

    def encrypt(self, plaintext):
        m = plaintext
        n = self.public_key
        assert 0 <= m < n

        # Choose a random r
        while True:
            r = generate_prime_number()
            if 0 < r < n and math.gcd(r, n) == 1:
                break
        # The ciphertext is given by
        # c = (n+1)^m * r^n * mod n^2
        c = (pow(n+1, m, n**2) * pow(r, n, n**2)) % n**2
        assert c <= n**2
        return c

    def decrypt(self, c):
        l = self.private_key[0]
        m = self.private_key[1]
        n = self.public_key

        # Define the integer division function
        # L(u) = (u - 1) / n
        def step1(u):
            return (u - 1) // n
        # The plaintext is
        # m = L(c^l mod n^2) * u mod n
        m = step1(pow(c, l, n**2)) * m % n
        return m

    def __generate_keys(self):
        while True:
            p = generate_prime_number()
            q = generate_prime_number()
            if p != q and math.gcd(p*q, (p-1)*(1-q)) == 1:
                break
        n = p * q

        l = (p-1) * (q-1)
        m = self.__invmod(l, n)
        return (n, (l, m))

    def __invmod(self, a, p, maxiter=1000000):
        """
        The multiplicitive inverse of a in the integers modulo p:
            a * b == 1 mod p
        Returns b.
        (http://code.activestate.com/recipes/576737-inverse-modulo-p/)
        """
        if a == 0:
            raise ValueError('0 has no inverse mod {}'.format(p))
        r = a
        d = 1
        for i in range(min(p, maxiter)):
            d = ((p // r + 1) * d) % p
            r = (d * a) % p
            if r == 1:
                break
        else:
            raise ValueError('{} has no inverse mod {}'.format(a, p))
        return d

class CryptoAlgo(Algorithm):
    def __init__(self, agents, time):
        self.agents = agents
        self.time = time
        self.keys = self.__get_keys(self.agents)
        self.__init_avg = self.__average(self.agents)
        self.__agents_history = []

    def run(self, log=False):
        # Algorithm from III. PROBLEM FORMULATION
        for k in range(self.time): # this is step 4
            self.__process(k)
        if log:
            prnt(self.__agents_history)

    def plot(self, show=False, save=False, tag="figure"):
        pass
    
    def __average(self, data):
        return sum(data) / len(data)

    def __get_keys(self, agents):
        return [SimplePaillier() for i in range(len(agents))]
    
    def __process(self, k):
        for i, v1 in enumerate(self.agents):
            v1_key = self.keys[i]
            for j, v2 in enumerate(self.agents):
                if v1 == v2: continue
                v2_key = self.keys[j]

                # Encrypt the Negative State
                v1, v2 = self.__step1(v1, v2, v1_key, v2_key)
                print(v1, v2)


    def __step1(self, v1, v2, v1_key, v2_key):
        # why minus? negative state
        return v1_key.encrypt(-v1), v2_key.encrypt(-v2)
        


if __name__ == "__main__":
    algo = CryptoAlgo([1, 2, 4, 8], 1)
    algo.run()
