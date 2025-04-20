import numpy as np
import random
import matplotlib.pyplot as plt


def dominates(a, b):
    return all(a <= b) and any(a < b)

class MOPSO:
    def __init__(self, pN, dim, max_iter, upper_bound, lower_bound, func, k=0.1, archive_size=100):
        self.w_max = 1.5
        self.w_min = 0.6
        self.c1 = 2.0
        self.c2 = 2.0
        self.k = k

        self.pN = pN
        self.dim = dim
        self.max_iter = max_iter
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.func = func  # func: [f1, f2, ..., fn]
        self.archive_size = archive_size  # External Archive Size

        self.X = np.zeros((self.pN, self.dim))
        self.Y = np.zeros((self.pN, self.dim))
        self.pBest = np.zeros((self.pN, self.dim))
        self.p_fitness = [None for _ in range(pN)]  # Fitness for fi for every p
        self.archive = []  # [x1,x2, ..., xn, fitness]

    def init(self):
        self.X = np.random.uniform(self.lower_bound, self.upper_bound, (self.pN, self.dim))
        v_max = self.k * (self.upper_bound - self.lower_bound)
        self.V = np.random.uniform(self.lower_bound, v_max, (self.pN, self.dim))
        self.pBest = self.X.copy()
        self.p_fitness = np.array([self.func(ind) for ind in self.pBest])
        self.archive = []

        for i in range(self.pN):
            self.update_archive(self.X[i], self.p_fitness[i])
        self.convergence_curve = []

    def update_archive(self, x, f):
        """Update Pareto Front ie. Archive"""
        dominated = []
        is_dominated = False
        for i, (xa, fa) in enumerate(self.archive):
            if dominates(f, fa):
                # f(new) dominate fa
                dominated.append(i)
            elif dominates(fa, f):
                # fa(old) dominate f
                is_dominated = True
                break

        if not is_dominated:
            for i in reversed(dominated):
                self.archive.pop(i)
            self.archive.append((x.copy(), f))

            if (len(self.archive) > self.archive_size):
                objs = [f for (_, f) in self.archive]
                dists = self.crowding_distance(objs)
                worst_idx = np.argmin(dists)
                self.archive.pop(worst_idx)

    def crowding_distance(self, archive_f):
        """
        Choose the archive By crowding distance
        :param archive_f:(N, M) -> Number of N pN, the number of Obj is M
        :return: (N) -> crowding distance of every object in the archive
        """
        archive_f = np.array(archive_f)
        N, M = np.shape(archive_f)
        distance = np.zeros(N)

        for m in range(M):
            # for f_{m} calculate distance
            sorted_idx = np.argsort(archive_f[:, m])
            f_min = archive_f[sorted_idx[0], m]
            f_max = archive_f[sorted_idx[-1], m]
            distance[sorted_idx[0]] = distance[sorted_idx[-1]] = np.inf
            for i in range(1, N - 1):
                if f_max - f_min == 0:
                    continue
                distance[sorted_idx[i]] += (archive_f[sorted_idx[i + 1], m] - archive_f[sorted_idx[i - 1], m]) / (f_max - f_min)
        return distance

    def select_leader(self):
        """Random choose the leader"""
        if len(self.archive) == 0:
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        return random.choice(self.archive)[0]

    def iterator(self):
        v_max = self.k * (self.upper_bound - self.lower_bound)
        for t in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * t / self.max_iter
            for i in range(self.pN):
                r1 = random.random()
                r2 = random.random()
                leader = self.select_leader()
                self.V[i] = (w * self.V[i] +
                             self.c1 * r1 * (self.pBest[i] - self.X[i]) +
                             self.c2 * r2 * (leader - self.X[i]))
                self.V[i] = np.clip(self.V[i], -v_max, v_max)
                self.X[i] += self.V[i]
                self.X[i] = np.clip(self.X[i], self.lower_bound, self.upper_bound)

                f = self.func(self.X[i])
                self.p_fitness[i] = f
                self.update_archive(self.X[i], f)

                if dominates(f, self.func(self.pBest[i])):
                    self.pBest[i] = self.X[i].copy()

            f1_vals = [f[0] for (_, f) in self.archive]
            f2_vals = [f[1] for (_, f) in self.archive]
            print(
                f"Iter {t + 1}/{self.max_iter}, Pareto size: {len(self.archive)}, Avg f1={np.mean(f1_vals):.4f}, f2={np.mean(f2_vals):.4f}")
            self.convergence_curve.append((np.mean(f1_vals), np.mean(f2_vals)))

        return self.archive

def test_func(x):
    # ZDT-1
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - np.sqrt(x[0] / g))
    return np.array([f1, f2])

if __name__ == "__main__":
    dim = 10
    mopso = MOPSO(pN=30, dim=dim, max_iter=100, upper_bound=1.0, lower_bound=0.0, func=test_func)
    mopso.init()
    pareto_archive = mopso.iterator()
    print("\nFinal Pareto Front (Objective Values):")
    for i, (x, f) in enumerate(pareto_archive):
        print(f"Solution {i + 1}: f = {f}, x = {x}")


f1_list = [f[0] for (_, f) in pareto_archive]
f2_list = [f[1] for (_, f) in pareto_archive]

plt.figure(figsize=(8, 6))
plt.scatter(f1_list, f2_list, c='red', label='Pareto Front')
plt.xlabel("f1")
plt.ylabel("f2")
plt.title("Final Pareto Front Approximated by MOPSO")
plt.legend()
plt.grid(True)
plt.show()




