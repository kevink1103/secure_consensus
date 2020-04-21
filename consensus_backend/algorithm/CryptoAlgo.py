# Secure and Privacy-Preserving Consensus
# by Minghao Ruan, Huan Gao, Student Member, IEEE, and Yongqiang Wang, Senior Member, IEEE
# DOI 10.1109/TAC.2019.2890887, IEEE Transactions on Automatic Control

import os
import time
import math
import random

import sympy
import numpy as np
import matplotlib.pyplot as plt
from gmpy2 import mpz, mpfr, is_prime
from pyprnt import prnt

try:
    from algorithm import Algorithm
except:
    from Algorithm import Algorithm

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
        m = int(plaintext)

        n = self.public_key
        assert 0 <= m < n

        # Choose a random r
        while True:
            r = self.__generate_prime_number()
            if 0 < r < n and math.gcd(r, n) == 1:
                break
        # The ciphertext is given by
        # c = (n+1)^m * r^n * mod n^2
        c = (pow(n+1, m, n**2) * pow(r, n, n**2)) % n**2
        assert 0 <= c < n**2
        assert math.gcd(c, n**2) == 1
        return c

    def decrypt(self, c):
        c = int(c)

        l, u = self.private_key
        n = self.public_key

        # Define the integer division function
        # L(u) = (u - 1) / n
        def L(u):
            return (u - 1) // n
        # The plaintext is
        # m = L(c^l mod n^2) * u mod n
        m = L(pow(c, l, n**2)) * u % n
        return m

    def __generate_prime_number(self, length=128):
        """ Generate a prime
            Args:
                length -- int -- length of the prime to generate, in bits 
            return a prime
        """
        p = 4
        # keep generating while the primality test fail
        while not is_prime(p, length):
            # generate an odd integer randomly
            p = random.getrandbits(length)
            # apply a mask to set MSB and LSB to 1
            p |= (1 << length - 1) | 1
        return p

    def __generate_keys(self):
        while True:
            p = self.__generate_prime_number()
            q = self.__generate_prime_number()
            #  if p != q and math.gcd(p*q, (p-1)*(1-q)) == 1:
            if p != q:
                break
        n = p * q

        l = (p-1) * (q-1)
        m = sympy.mod_inverse(l, n)
        return (n, (l, m))

class CryptoAlgo(Algorithm):
    def __init__(self, topology, epsilon, time):
        self.Q = 10 ** 4
        self.A = topology.A
        self.agents = [agent * self.Q for agent in topology.lagents]
        self.epsilon = epsilon if 0 < epsilon < 1 else 0.5
        self.time = time
        self.keys = self.__get_keys(self.agents)
        self.tag = "{}_ε{}".format(topology.name, epsilon)
        self.__init_avg = self.__average(self.agents) / self.Q
        self.__agents_history = [self.agents]
        self.__ciphertext_history = []
        self.__consensus = 0

    def run(self, log=False):
        # Algorithm from III. PROBLEM FORMULATION
        for k in range(self.time): # this is step 4
            self.__process(k)
        if log:
            prnt(self.__agents_history)

    def plot(self, show=False, save=False):
        if len(self.__agents_history) == 0:
            raise AssertionError("empty data to plot")

        # reorganize
        ys = np.array(self.__agents_history).T / self.Q
        print(ys)
        # error vector
        zs = np.array(self.__ciphertext_history).T
        print(zs)
        #  print(exps)
        #  zs = ys
        #  raise

        # PLOT
        title = "crypto_{}".format(self.tag)
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
        #  plt.ylim([-2, 10])
        plt.legend()
        # chart 2
        plt.subplot(size[0], size[1], 2)
        plt.title('Encrypted Diff. State Vector')
        x = np.arange(0, self.time, 1)
        #  for i, z in enumerate(zs):
        #      plt.plot(x, z, label='z{}(k)'.format(i+1))
        plt.plot(x, zs[0], label='ε1(a2->1(x2[k]-x1[k]))')
        plt.plot(x, zs[1], label='ε2(a3->2(x2[k]-x1[k]))')
        plt.plot(x, zs[2], label='ε3(a4->3(x2[k]-x1[k]))')
        plt.plot(x, zs[3], label='ε4(a1->4(x2[k]-x1[k]))')
        plt.axhline(y=0, color='k', linestyle='dashed')
        plt.plot(self.__consensus, 0, marker='x', markersize=10, color="black")
        plt.xlabel('k')
        plt.ylabel('log(εi(·))')
        plt.xlim([0, self.time])
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

    def __get_keys(self, agents):
        return [SimplePaillier() for i in range(len(agents))]

    def __process(self, k):
        # delta =  max(i)|N(i)|
        delta = max([ len([val for j, val in enumerate(a) if i != j and val > 0]) for i, a in enumerate(self.A.tolist()) ])

        weighted_diffs = np.zeros(self.A.shape).tolist()
        ciphertexts = []
        new_states = []
        timestamps = []

        for i, v1 in enumerate(self.agents):

            v1_key = self.keys[i]

            a_lower, a_upper = self.__generate_admissible_range(delta)

            for j, v2 in enumerate(self.agents):
                if i == j:
                    continue

                timestamp = time.time()

                v2_key = self.keys[j]

                #  prnt("begin", [v1, v2])
                #  prnt([i, j])

                if weighted_diffs[i][j] != 0:
                    #  print("weight exists, skip this")
                    continue

                aij = self.A[i, j]
                ai_j, aj_i = self.__generate_weight(k, a_lower, a_upper, aij)
                ai_j, aj_i = int(round(ai_j * self.Q)), int(round(aj_i * self.Q))

                # Step 1
                # Encrypt the Negative State (with its own key)
                step1_v1, step1_v2 = self.__step1(k, v1, v2, v1_key, v2_key)
                #  prnt("step1", [step1_v1, step1_v2])

                # Step 2
                # Transmit the State and Public Key
                step2_v1, step2_v2 = self.__step2(k, step1_v1, step1_v2)
                #  prnt("step2", [step2_v1, step2_v2])

                # Step 3
                # Encrypt the State (with received key)
                step3_v1, step3_v2 = self.__step3(k, v1, v2, v1_key, v2_key)
                #  prnt("step3", [step3_v1, step3_v2])

                # Step 4
                # Compute the Difference (in ciphertext)
                step4_v1, step4_v2 = self.__step4(k, step3_v1, step2_v1, step3_v2, step2_v2)
                #  prnt("step4", [step4_v1, step4_v2])
                #  prnt([self.__decrypt_mul(v2_key, step4_v1), self.__decrypt_mul(v1_key, step4_v2)])

                # Step 5
                # Multiply the Weight (in ciphertext)
                step5_v1, step5_v2 = self.__step5(k, step4_v1, ai_j, step4_v2, aj_i)
                #  prnt("step5", [step5_v1, step5_v2])

                # Step 6
                # Trasmit the Result Back to Sender
                step6_v1, step6_v2 = self.__step6(k, step5_v1, step5_v2)
                #  prnt("step6", [step6_v1, step6_v2])

                # Record
                if (i == 0 and j == 1):
                    ciphertexts.append(math.log(step6_v1))
                if (i == 1 and j == 2):
                    ciphertexts.append(math.log(step6_v1))
                if (i == 2 and j == 3):
                    ciphertexts.append(math.log(step6_v1))
                if (i == 0 and j == 4):
                    ciphertexts.append(math.log(step6_v2))

                # Step 7
                # Decrypt the Result
                step7_v1, step7_v2 = self.__step7(k, step6_v1, step6_v2, v1_key, v2_key)
                #  prnt("step7", [step7_v1, step7_v2])

                # Step 8
                # Multiply the Weight (in plaintext)
                step8_v1, step8_v2 = self.__step8(k, ai_j, step7_v1, aj_i, step7_v2)
                #  prnt("step8", [step8_v1, step8_v2])

                # Scale Down Weight from ai_j and aj_i
                weight_v1 = mpz(step8_v1) / mpz(self.Q) / mpz(self.Q)
                weight_v2 = mpz(step8_v2) / mpz(self.Q) / mpz(self.Q)

                #  prnt([ai_j, aj_i])

                #  print("WEIGHT")
                #  print(weight_v1, weight_v2)
                #  print()

                weighted_diffs[i][j] = int(weight_v1)
                weighted_diffs[j][i] = int(weight_v2)

                #  print("ONE EXCHANGE TOOK", time.time()-timestamp, "sec")
                timestamps.append(time.time() - timestamp)

            # Update Rule (3)
            #  print(weighted_diffs)
            #  print(np.asarray(weighted_diffs, dtype=np.float32))
            #  print()

            weight_sum = mpz(0)
            for n in weighted_diffs[i]:
                weight_sum += mpz(n)

            new_state = mpz(v1) + (mpfr(self.epsilon) * weight_sum)
            new_state = int(round(new_state))
            new_states.append(new_state)
            #  print(new_state)
        #  print(k)
        prnt(new_states)
        self.agents = new_states
        self.__agents_history.append(new_states)
        self.__ciphertext_history.append(ciphertexts)
        avg_timestamp = sum(timestamps) / len(timestamps)
        print("AVERAGE EXCHANGE TIME:", avg_timestamp, "sec")

        # check if consensus is made
        new_states = [float("{0:.1f}".format(agent / self.Q)) for agent in new_states]
        agents_average = float("{0:.1f}".format(self.__init_avg))
        if all(agent-agents_average == 0 for agent in new_states):
            self.__consensus = k if self.__consensus == 0 else self.__consensus

    def __step1(self, k, v1, v2, v1_key, v2_key):
        neg_v1 = self.__twos_complement(-v1) if v1 > 0 else v1
        neg_v2 = self.__twos_complement(-v2) if v2 > 0 else v2
        step1_v1 = v1_key.encrypt(neg_v1)
        step1_v2 = v2_key.encrypt(neg_v2)
        return step1_v1, step1_v2

    def __step2(self, k, step1_v1, step1_v2):
        step2_v1 = step1_v2
        step2_v2 = step1_v1
        return step2_v1, step2_v2

    def __step3(self, k, v1, v2, v1_key, v2_key):
        step3_v1 = v2_key.encrypt(v1)
        step3_v2 = v1_key.encrypt(v2)
        return step3_v1, step3_v2

    def __step4(self, k, step3_v1, step2_v1, step3_v2, step2_v2):
        step4_v1 = step3_v1 * step2_v1
        step4_v2 = step3_v2 * step2_v2
        return step4_v1, step4_v2

    def __step5(self, k, step4_v1, ai_j, step4_v2, aj_i):
        step5_v1 = mpz(step4_v1) ** mpz(ai_j)
        step5_v2 = mpz(step4_v2) ** mpz(aj_i)
        step5_v1, step5_v2 = int(step5_v1), int(step5_v2)
        return step5_v1, step5_v2

    def __step6(self, k, step5_v1, step5_v2):
        step6_v1 = step5_v2
        step6_v2 = step5_v1
        return step6_v1, step6_v2

    def __step7(self, k, step6_v1, step6_v2, v1_key, v2_key):
        step7_v1 = self.__decrypt_mul(v1_key, step6_v1)
        step7_v2 = self.__decrypt_mul(v2_key, step6_v2)
        return step7_v1, step7_v2

    def __step8(self, k, ai_j, step7_v1, aj_i, step7_v2):
        step8_v1 = ai_j * step7_v1
        step8_v2 = aj_i * step7_v2
        return step8_v1, step8_v2

    def __twos_complement(self, val):
        nbits = 64

        if val < 0:
            val = (1 << nbits) + val
        else:
            bin_string = ""
            for c in np.binary_repr(val)[-nbits:]:
                if c == "0":
                    bin_string += "1"
                else:
                    bin_string += "0"
            val = int(bin_string, 2) + 1
            val *= -1
        return val

    def __decrypt_mul(self, key, mul):
        ans = key.decrypt(mul)
        bin_ans = np.binary_repr(ans)

        # negative if 64th bit is 1
        neg = False
        if len(bin_ans) >= 64:
            neg = bin_ans[::-1][64-1] == "1"

        # discard overflown bits
        ans = int(bin_ans[-64:], 2)

        # apply negative
        if neg:
            ans = self.__twos_complement(ans)
        return ans

    def __generate_admissible_range(self, delta):
        max_range = 1 / (math.sqrt(self.epsilon * delta))

        while True:
            # random.uniform() returns random from [a, b]
            a_lower = random.uniform(0, max_range)
            a_upper = random.uniform(0, max_range)
            if 0 < a_lower < a_upper < max_range:
                return a_lower, a_upper

    def __generate_weight(self, k, a_lower, a_upper, aij):
        if aij == 0:
            return 0, 0
        while True:
            if k == 0:
                ai_j = random.random()
                aj_i = aij / ai_j
            else:
                ai_j = random.uniform(a_lower, a_upper)
                aj_i = aij / ai_j
            if ai_j > 0 and aj_i > 0:
                return ai_j, aj_i


if __name__ == "__main__":
    def twos_complement(val):
        nbits = 64

        if val < 0:
            val = (1 << nbits) + val
        else:
            new_string = ""
            for c in np.binary_repr(val):
                if c == "0":
                    new_string += "1"
                else:
                    new_string += "0"
            val = int(new_string[-nbits:], 2) + 1
            val *= -1
        return val

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
    print("DONE")
    print(twos_complement(m))

    print()

    print("HERE WE GO")

    def process(a, b):
        a = twos_complement(a) if a < 0 else a
        b = twos_complement(b) if b < 0 else b

        Q = 10 ** 3

        a, b = a*Q, b*Q

        print("START")

        print("m", a, b)

        e_a = key.encrypt(a)
        e_b = key.encrypt(b)
        print("dm", key.decrypt(e_a), key.decrypt(e_b))

        mul = e_a * e_b
        print(e_a)
        print(mul)
        print()

        ans = key.decrypt(mul)
        print("NO MUL ANS", ans)
        bb = np.binary_repr(ans)
        print(bb)
        print(len(bb))
        print(bb[-64:])
        print(len(bb[-64:]))
        print()

        t = mul ** int(0.25 * Q)
        print("LEN", len(str(t)))
        ans = key.decrypt(mul)
        print("MUL ORI ANS", ans)
        bb = np.binary_repr(ans)
        print(bb)
        print(len(bb))
        print(bb[-64:])
        print(len(bb[-64:]))
        if len(bb) >= 64:
            ans = twos_complement(ans)
        print()
        return int(ans / int(Q))

    #  for a, b, c in [[6, 2, 8], [6, -8, -2], [-12, 6, -6], [-2, -5, -7]]:
    for a, b, c in [[-2, -5, -7]]:
        ans = process(a, b)
        print("INIT", a, b)
        print("RESULT", ans, c)
        bb = np.binary_repr(ans)
        print("BIN", bb)
        print("LEN", len(bb))
        break

