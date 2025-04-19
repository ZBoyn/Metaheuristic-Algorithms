import numpy as np

def get_benchmark_function(name):
    functions = {
        'F1': {
            'func': F1,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Sphere',
            'range': '[-100, 100]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F2': {
            'func': F2,
            'lb': -10,
            'ub': 10,
            'dim': 30,
            'name': 'Schwefel 2.22',
            'range': '[-10, 10]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F3': {
            'func': F3,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Schwefel 1.2',
            'range': '[-100, 100]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F4': {
            'func': F4,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Schwefel 2.21',
            'range': '[-100, 100]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F5': {
            'func': F5,
            'lb': -30,
            'ub': 30,
            'dim': 30,
            'name': 'Rosenbrock',
            'range': '[-30, 30]',
            'minimum': 'f(x*)=0 at x*={1, 1, ..., 1}'
        },
        'F6': {
            'func': F6,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'step',
            'range': '[-100, 100]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F7': {
            'func': F7,
            'lb': -1.28,
            'ub': 1.28,
            'dim': 30,
            'name': 'Quartic',
            'range': '[-1.28, 1.28]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F8': {
            'func': F8,
            'lb': -1,
            'ub': 1,
            'dim': 30,
            'name': 'Sum of Different Powers',
            'range': 'range: [-10, 10]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F9': {
            'func': F9,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'High Conditioned Elliptic',
            'range': '[-100, 100]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F10': {
            'func': F10,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Discus',
            'range': '[-100, 100]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F11': {
            'func': F11,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Bent Cigar',
            'range': '[-100, 100]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F12': {
            'func': F12,
            'lb': -5,
            'ub': 10,
            'dim': 30,
            'name': 'Zakharov',
            'range': '[-5, 10]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F13': {
            'func': F13,
            'lb': -1,
            'ub': 1,
            'dim': 30,
            'name': 'Tablet',
            'range': '[-1, 1]^D',
            'minimum': 'f(x*)=0'
        },
        'F14': {
            'func': F14,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Chung Reynolds',
            'range': '[-100, 100]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F15': {
            'func': F15,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Sum squares',
            'range': '[-10, 10]^D',
            'minimum': 'f(x*)=0'
        },
        'F16': {
            'func': F16,
            'lb': -500,
            'ub': 500,
            'dim': 30,
            'name': 'Schwefel 2.26',
            'range': '[-500, 500]',
            'minimum': 'f(x*)=-418.9829*D at x*={420, 420, ..., 420}'
        },
        'F17': {
            'func': F17,
            'lb': -5.12,
            'ub': 5.12,
            'dim': 30,
            'name': 'Rastrigin',
            'range': '[-5.12, 5.12]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F18': {
            'func': F18,
            'lb': -32,
            'ub': 32,
            'dim': 30,
            'name': 'Ackley',
            'range': '[-32, 32]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F19': {
            'func': F19,
            'lb': -600,
            'ub': 600,
            'dim': 30,
            'name': 'Griewank',
            'range': '[-600, 600]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F20': {
            'func': F20,
            'lb': -50,
            'ub': 50,
            'dim': 30,
            'name': 'Penalized',
            'range': '[-50, 50]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F21': {
            'func': F21,
            'lb': -50,
            'ub': 50,
            'dim': 30,
            'name': 'Penalized2',
            'range': '[-50, 50]',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F22': {
            'func': F22,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Schaffer',
            'range': '[-100, 100]^D',
            'minimum': 'f(x*)=0'
        },
        'F23': {
            'func': F23,
            'lb': -10,
            'ub': 10,
            'dim': 30,
            'name': 'Alpine',
            'range': '[-10, 10]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F24': {
            'func': F24,
            'lb': -1,
            'ub': 1,
            'dim': 30,
            'name': 'Csendes',
            'range': '[-1, 1]^D',
            'minimum': 'f(x*)=0'
        },
        'F25': {
            'func': F25,
            'lb': -1,
            'ub': 1,
            'dim': 30,
            'name': 'Inverted Cosine Mixture',
            'range': '[-1, 1]^D',
            'minimum': 'f(x*)=0'
        },
        'F26': {
            'func': F26,
            'lb': -15,
            'ub': 15,
            'dim': 30,
            'name': 'Bohachevsky',
            'range': '[-5, 5]^D',
            'minimum': 'f(x*)=0'
        },
        'F27': {
            'func': F27,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Schaffer’s F7',
            'range': '[-100, 100]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F28': {
            'func': F28,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Expanded Schaffer’s F6',
            'range': '[-100, 100]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F29': {
            'func': F29,
            'lb': -100,
            'ub': 100,
            'dim': 30,
            'name': 'Salomon',
            'range': '[-20, 20]^D',
            'minimum': 'f(x*)=0 at x*={0, 0, ..., 0}'
        },
        'F30': {
            'func': F30,
            'lb': -10,
            'ub': 10,
            'dim': 30,
            'name': 'Quintic',
            'range': '[-10,10]',
            'minimum': 'f(x*)=0 at x*={-1, -1, ..., -1} or x*={2, 2, ..., 2}'
        },
    }

    return functions[name]

# Benchmark function definitions
def F1(x):
    return np.sum(np.square(x))

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    return np.sum([np.sum(x[:i + 1]) ** 2 for i in range(len(x))])

def F4(x):
    return np.max(np.abs(x))

def F5(x):
    x = np.asarray(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

def F6(x):
    return np.sum(np.square(np.abs(x + 0.5)))

def F7(x):
    dim = len(x)
    return np.sum([(i + 1) * xi ** 4 for i, xi in enumerate(x)]) + np.random.rand()

def F8(x):
    D = len(x)
    powers = np.arange(2, D + 2)
    return np.sum(np.abs(x) ** powers)

def F9(x):
    D = len(x)
    coeffs = (10 ** 6) ** (np.arange(D) / (D - 1))
    return np.sum(coeffs * np.square(x))

def F10(x):
    return 10 ** 6 * x[0] ** 2 + np.sum(np.square(x[1:]))

def F11(x):
    return x[0] ** 2 + 10 ** 6 * np.sum(np.power(x[1:], 6))

def F12(x):
    D = len(x)
    idx = np.arange(1, D + 1)
    sum1 = np.sum(np.square(x))
    sum2 = np.sum(0.5 * idx * x)
    return sum1 + sum2 ** 2 + sum2 ** 4

def F13(x):
    return 10 ** 6 * x[0] ** 2 + np.sum(np.power(x[1:], 6))

def F14(x):
    return np.sum(np.square(x)) ** 2

def F15(x):
    idx = np.arange(1, len(x) + 1)
    return np.sum(idx * np.square(x))

def F16(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F17(x):
    x = np.asarray(x)
    return np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x)) + 10 * len(x)

def F18(x):
    x = np.asarray(x)
    dim = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(x)) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.e

def F19(x):
    x = np.asarray(x)
    return np.sum(np.square(x)) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def Ufun(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)

def F20(x):
    x = np.asarray(x)
    dim = len(x)
    term1 = 10 * np.sin(np.pi * (1 + (x[0] + 1) / 4)) ** 2
    term2 = np.sum(((x[:-1] + 1) / 4) ** 2 * (1 + 10 * np.sin(np.pi * (1 + (x[1:] + 1) / 4)) ** 2))
    term3 = ((x[-1] + 1) / 4) ** 2
    penalty = np.sum(Ufun(x, 10, 100, 4))
    return (np.pi / dim) * (term1 + term2 + term3) + penalty

def F21(x):
    x = np.asarray(x)
    dim = len(x)
    term1 = np.sin(3 * np.pi * x[0]) ** 2
    term2 = np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
    term3 = (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)
    penalty = np.sum(Ufun(x, 5, 100, 4))
    return 0.1 * (term1 + term2 + term3) + penalty

def F22(x):
    sum1 = np.sum(np.square(x))
    return 0.5 + (np.sin(np.sqrt(sum1)) ** 2 - 0.5) / (1 + 0.001 * sum1) ** 2

def F23(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

def F24(x):
    return np.sum(x ** 6 * (2 + np.sin(1 / x)))

def F25(x):
    D = len(x)
    return 0.1 * D - (0.1 * np.sum(np.cos(5 * np.pi * x)) - np.sum(x ** 2))

def F26(x):
    x1 = x[:-1]
    x2 = x[1:]
    return np.sum(x1 ** 2 + 2 * x2 ** 2 - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7)

def F27(x):
    x = np.asarray(x)
    si = np.sqrt(x[:-1] ** 2 + x[1:] ** 2)
    return (1 / (len(x) - 1) * np.sum(np.sqrt(si) * (np.sin(50 * si ** 2) + 1))) ** 2

def F28(x):
    def g(z, y):
        return 0.5 + (np.sin(np.sqrt(z ** 2 + y ** 2)) ** 2 - 0.5) / (1 + 0.001 * (z ** 2 + y ** 2)) ** 2
    return np.sum([g(x[i], x[i + 1]) for i in range(len(x) - 1)]) + g(x[-1], x[0])

def F29(x):
    sum1 = np.sum(np.square(x))
    return 1 - np.cos(2 * np.pi * np.sqrt(sum1)) + 0.1 * np.sqrt(sum1)

def F30(x):
    return np.sum(np.abs(x ** 5 - 3 * x ** 4 + 4 * x ** 3 + 2 * x ** 2 - 10 * x - 4))
