"""Benchmark functions (Surjanovic & Bingham, SFU). All minimise to 0."""
import numpy as np


def sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x):
    x = np.asarray(x, dtype=float)
    d = x.size
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / d)
    return float(term1 + term2 + 20.0 + np.e)


# name -> (function, lower bound, upper bound, known minimiser value per coordinate)
FUNCTIONS = {
    "sphere":     (sphere,     -5.12,   5.12,   0.0),
    "rosenbrock": (rosenbrock, -5.0,    10.0,   1.0),
    "rastrigin":  (rastrigin,  -5.12,   5.12,   0.0),
    "ackley":     (ackley,     -32.768, 32.768, 0.0),
}