import numpy as np
import random

class PSO():
    def __init__(self, pN, dim, max_iter, upper_bound, lower_bound, func, k=0.1,):
        self.w_max = 1.5
        self.w_min = 0.6
        self.c1 = 2
        self.c2 = 2
        self.k = k
        self.func = func
        self.pN = pN  # Number of particles
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.max_iter = max_iter
        self.X = np.zeros((self.pN, self.dim))
        self.V = np.zeros((self.pN, self.dim))
        self.pBest = np.zeros((self.pN, self.dim))
        self.gBest = np.zeros((1, self.dim))
        self.p_fitness = np.full(self.pN, np.inf)
        self.fit = np.inf
    
    def init(self):
        self.X = np.random.uniform(self.lower_bound, self.upper_bound, (self.pN, self.dim))
        v_max = self.k * (self.upper_bound - self.lower_bound)
        self.V = np.random.uniform(-v_max, v_max, (self.pN, self.dim))
        self.pBest = self.X.copy()
        self.p_fitness = np.array([self.func(ind) for ind in self.pBest])
        min_index = np.argmin(self.p_fitness)
        self.gBest = self.pBest[min_index].copy()
        self.fit = self.p_fitness[min_index]
        self.convergence_curve = []

    def iterator(self):
        v_max = self.k * (self.upper_bound - self.lower_bound)
        for t in range(self.max_iter):
            # Linear decreasing inertia weight
            w = self.w_max - (self.w_max - self.w_min) * t / self.max_iter
            for i in range(self.pN):
                r1 = random.random()
                r2 = random.random()
                # Update V
                self.V[i] = (w * self.V[i] +
                             self.c1 * r1 * (self.pBest[i] - self.X[i]) +
                             self.c2 * r2 * (self.gBest - self.X[i]))
                self.V[i] = np.clip(self.V[i], -v_max, v_max)
                # Update X
                self.X[i] += self.V[i]
                self.X[i] = np.clip(self.X[i], self.lower_bound, self.upper_bound)
                fitness = self.func(self.X[i])
                # Update pBest
                if fitness < self.p_fitness[i]:
                    self.p_fitness[i] = fitness
                    self.pBest[i] = self.X[i].copy()
                # Update gBest
                if fitness < self.fit:
                    self.fit = fitness
                    self.gBest = self.X[i].copy()
            
            print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.fit}")
            self.convergence_curve.append(self.fit)
        return self.gBest, self.fit
                