# Secure and Privacy-Preserving Consensus
# by Minghao Ruan, Huan Gao, Student Member, IEEE, and Yongqiang Wang, Senior Member, IEEE
# DOI 10.1109/TAC.2019.2890887, IEEE Transactions on Automatic Control

import os
import math
import random
from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt
from pyprnt import prnt

try:
    from algorithm import Algorithm
    from algorithm.Prime import generate_prime_number
except:
    from Algorithm import Algorithm
    from Prime import generate_prime_number

class SimplePaillier:
    def __init__(self):
        keys = self.__generate_keys()
        self.public_key = keys[0]
        self.private_key = keys[1]

    def __str__(self):
        result = []
        result.append("===PUBLIC KEY===")
        result.append(str(self.public_key))
        result.append("===PRIVATE KEY===")
        result.append("l: {}".format(self.private_key[0]))
        result.append("m: {}".format(self.private_key[1]))
        return "\n".join(result)

    def encrypt(self, plaintext):
        m = plaintext
        # consider shifting to the m (plaintext)

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
        assert 0 <= c < n**2
        assert math.gcd(c, n**2) == 1
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
    def __init__(self, A, agents, epsilon, time):
        self.A = A
        self.agents = agents
        self.epsilon = epsilon
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
                if v1 == v2:
                    continue

                v2_key = self.keys[j]

                ai_j, aj_i = self.__generate_weight(k, self.A[i, j])

                prnt("begin", [v1, v2])

                # Step 1
                # Encrypt the Negative State (with its own key)
                step1_v1, step1_v2 = self.__step1(k, v1, v2, v1_key, v2_key)
                prnt("step1", [step1_v1, step1_v2])

                # Step 2
                # Transmit the State and Public Key
                step2_v1, step2_v2 = self.__step2(k, step1_v1, step1_v2)
                prnt("step2", [step2_v1, step2_v2])

                # Step 3
                # Encrypt the State (with received key)
                step3_v1, step3_v2 = self.__step3(k, v1, v2, v1_key, v2_key)
                prnt("step3", [step3_v1, step3_v2])

                # Step 4
                # Compute the Difference (in ciphertext)
                step4_v1, step4_v2 = self.__step4(k, step3_v1, step2_v1, step3_v2, step2_v2)
                prnt("step4", [step4_v1, step4_v2])

                # Step 5
                # Multiply the Weight (in ciphertext)
                # Result becomes Decimal from here
                step5_v1, step5_v2 = self.__step5(k, step4_v1, ai_j, step4_v2, aj_i)
                prnt("step5", [step5_v1, step5_v2])

                # Step 6
                # Trasmit the Result Back to Sender
                step6_v1, step6_v2 = self.__step6(k, step5_v1, step5_v2)
                prnt("step6", [step6_v1, step6_v2])

                # Step 7
                # Decrypt the Result
                step7_v1, step7_v2 = self.__step7(k, step6_v1, step6_v2, v1_key, v2_key)
                prnt("step7", [step7_v1, step7_v2])

                # Step 8
                # Multiply the Weight (in plaintext)
                step8_v1, step8_v2 = self.__step8(k, ai_j, step7_v1, aj_i, step7_v2)
                prnt("step8", [step8_v1, step8_v2])

                break
            break

    def __step1(self, k, v1, v2, v1_key, v2_key):
        def twos_complement(val, nbits):
            """Compute the 2's complement of int value val"""
            if val < 0:
                val = (1 << nbits) + val
            else:
                if (val & (1 << (nbits - 1))) != 0:
                    # If sign bit is set.
                    # compute negative value.
                    val = val - (1 << nbits)
            return abs(val)

        # Apply two's complement
        v1 = twos_complement(-v1, len(bin(v1))-2)
        v2 = twos_complement(-v2, len(bin(v2))-2)

        return v1_key.encrypt(v1), v2_key.encrypt(v2)

    def __step2(self, k, step1_v1, step1_v2):
        return step1_v2, step1_v1

    def __step3(self, k, v1, v2, v1_key, v2_key):
        return v2_key.encrypt(v1), v1_key.encrypt(v2)

    def __step4(self, k, step3_v1, step2_v1, step3_v2, step2_v2):
        return step3_v1 * step2_v1, step3_v2 * step2_v2

    def __step5(self, k, step4_v1, ai_j, step4_v2, aj_i):
        print(step4_v1, ai_j)
        print(step4_v2, aj_i)
        return int(Decimal(step4_v1) ** Decimal(ai_j)), int(Decimal(step4_v2) ** Decimal(aj_i))
        #  return Decimal(step4_v1) ** Decimal(ai_j), Decimal(step4_v2) ** Decimal(aj_i)

    def __step6(self, k, step5_v1, step5_v2):
        return step5_v2, step5_v1

    def __step7(self, k, step6_v1, step6_v2, v1_key, v2_key):
        return v1_key.decrypt(step6_v1), v2_key.decrypt(step6_v2)

    def __step8(self, k, ai_j, step7_v1, aj_i, step7_v2):
        return ai_j * step7_v1, aj_i * step7_v2

    def __generate_weight(self, k, aij):
        if k == 0:
            ai_j = random.random()
            aj_i = aij / ai_j
            return ai_j, aj_i

        # get admissible range
        a_lower = None
        a_upper = None
        delta = 1
        max_range = 1 / (math.sqrt(self.epsilon) * delta)

        while True:
            # random.uniform() returns random from [a, b]
            a_lower = random.uniform(0, max_range)
            a_upper = random.uniform(0, max_range)
            if 0 < a_lower and a_lower < a_upper:
                break

        print("max", max_range)
        print(a_lower, a_upper)
        ai_j = random.uniform(a_lower, a_upper)
        aj_i = aij / ai_j
        print(ai_j, aj_i)

        return ai_j, aj_i


if __name__ == "__main__":
    def twos_complement(val):
        """Compute the 2's complement of int value val"""
        if val < 0:
            val = (1 << (len(bin(val))-3)) + val
        return val

    # algo = CryptoAlgo([1, 2, 4, 8], 1)
    # algo.run()
    key = SimplePaillier()
    print(key)

    m = -12
    m = twos_complement(m)
    print("===ENCRYPT: {}===".format(m))
    c = key.encrypt(m)
    print(c)
    print("===DECRYPT: {}===".format(c))
    m = key.decrypt(c)
    print(m)

