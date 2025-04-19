import numpy as np
from pso import PSO
import matplotlib.pyplot as plt
from benchmark import *
from plot import plot_benchmark_function

if __name__ == "__main__":
    dim = 30
    F = "F11"
    pso = PSO(pN=30, dim=dim, max_iter=100, upper_bound=100, lower_bound=-100, func=F11)
    pso.init()
    gbest, best_fitness = pso.iterator()
    print("Best position:", gbest)
    print("Best fitness:", best_fitness)

    plt.plot(range(1, pso.max_iter + 1), pso.convergence_curve)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Curve of PSO')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    info = get_benchmark_function(F)
    print("Function name:", info['name'])
    print("range", info['range'])
    print("minimum:", info['minimum'])
    plot_benchmark_function(F)