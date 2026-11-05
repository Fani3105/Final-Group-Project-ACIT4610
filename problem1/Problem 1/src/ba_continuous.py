"""Bees Algorithm for continuous minimisation (Part B).

One iteration:
  1. Sort the population; the best m bees are the selected sites,
     the best e of them are elite sites.
  2. Recruit nep neighbours around each elite site and nsp around each
     other selected site. A neighbour changes every coordinate by an
     independent uniform draw in [-0.1, +0.1] x (upper - lower), then clips.
  3. Each site keeps the best of itself and its neighbours.
  4. The other n - m bees are replaced by fresh uniform random scouts.
Every call to the objective counts against the budget (initial population,
neighbours, scouts, rejected candidates). The run stops the moment the
budget is reached, so the last iteration may be incomplete.
"""
import time

import numpy as np

DEFAULT_SETTINGS = {
    "n": 30,      # scouts / population size
    "m": 10,      # selected sites
    "e": 3,       # elite sites (subset of m)
    "nep": 10,    # recruited bees per elite site
    "nsp": 5,     # recruited bees per non-elite selected site
    "ngh": 0.10,  # neighbourhood half-width as a fraction of (upper - lower)
}
DEFAULT_BUDGET = 30_000


class BudgetExhausted(Exception):
    pass


class Evaluator:
    """Wraps the objective: counts evaluations, tracks best-so-far."""

    def __init__(self, func, budget):
        self.func = func
        self.budget = budget
        self.evals = 0
        self.best_f = np.inf
        self.best_x = None
        self.history = []  # best-so-far after each evaluation

    def __call__(self, x):
        if self.evals >= self.budget:
            raise BudgetExhausted
        fx = self.func(x)
        self.evals += 1
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        self.history.append(self.best_f)
        return fx


def run_ba_continuous(func, lower, upper, dim, seed,
                      settings=None, budget=DEFAULT_BUDGET):
    s = dict(DEFAULT_SETTINGS)
    if settings:
        s.update(settings)
    n, m, e, nep, nsp, ngh = s["n"], s["m"], s["e"], s["nep"], s["nsp"], s["ngh"]
    assert 0 < e <= m <= n

    rng = np.random.default_rng(seed)
    lo = np.full(dim, float(lower))
    hi = np.full(dim, float(upper))
    width = hi - lo
    ev = Evaluator(func, budget)

    def random_bee():
        return rng.uniform(lo, hi)

    def neighbour(x):
        step = rng.uniform(-ngh, ngh, size=dim) * width
        return np.clip(x + step, lo, hi)

    t0 = time.perf_counter()
    initial_best = None
    try:
        # initial population (counted)
        pop = [random_bee() for _ in range(n)]
        fit = [ev(x) for x in pop]
        initial_best = ev.best_f

        while True:
            order = np.argsort(fit)
            sites = [(pop[i], fit[i]) for i in order[:m]]
            new_sites = []
            for rank, (x, fx) in enumerate(sites):
                k = nep if rank < e else nsp
                best_x, best_fx = x, fx
                for _ in range(k):
                    cand = neighbour(x)
                    fc = ev(cand)          # counted even if rejected
                    if fc < best_fx:
                        best_x, best_fx = cand, fc
                new_sites.append((best_x, best_fx))
            # fresh scouts for the remaining n - m bees (counted)
            scouts = []
            for _ in range(n - m):
                x = random_bee()
                scouts.append((x, ev(x)))
            nxt = new_sites + scouts
            pop = [p[0] for p in nxt]
            fit = [p[1] for p in nxt]
    except BudgetExhausted:
        pass
    runtime = time.perf_counter() - t0

    return {
        "seed": seed,
        "best_f": float(ev.best_f),
        "best_x": ev.best_x,
        "initial_best": float(initial_best),
        "history": np.array(ev.history),  # length == evals used
        "evals": ev.evals,
        "runtime": runtime,
        "settings": s,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from functions import FUNCTIONS

    for name, (f, lo, hi, _) in FUNCTIONS.items():
        for D in (10, 30):
            r = run_ba_continuous(f, lo, hi, D, seed=1)
            print(f"{name:11s} D={D:2d}  initial={r['initial_best']:.4g}  "
                  f"final={r['best_f']:.4g}  evals={r['evals']}  "
                  f"time={r['runtime']:.2f}s")