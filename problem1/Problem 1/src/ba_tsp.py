cat > src/ba_tsp.py << 'EOF'
"""Bees Algorithm for the symmetric TSP (Part A).

Same loop as ba_continuous.py; only the representation and the neighbour
move differ.
  - A solution is a permutation of 0..n-1 (0-based city indices) with city 0
    (TSPLIB city 1) fixed in position 0.
  - move="swap":    exchange the cities at two random positions (1..n-1).
  - move="reverse": reverse the segment between two random positions (1..n-1).
  Each recruited neighbour is created by ONE random move. Both moves keep
  the permutation valid and never touch position 0.
Every objective call (initial population, neighbours, scouts, including
rejected ones) counts against the budget; the run stops at the budget.
"""
import time

import numpy as np

from ba_continuous import BudgetExhausted, Evaluator
from tsp_loader import tour_length, validate_tour

DEFAULT_SETTINGS = {
    "n": 10,     # scouts / population size
    "m": 3,      # selected sites
    "e": 1,      # elite sites
    "nep": 20,   # recruited neighbours per elite site
    "nsp": 10,   # recruited neighbours per other selected site
}
DEFAULT_BUDGET = 30_000

# TSPLIB optimal tour lengths (used only to compute the gap afterwards)
OPTIMA = {"eil51": 426, "berlin52": 7542, "st70": 675,
          "eil76": 538, "kroA100": 21282, "ch150": 6528}


def swap_move(tour, rng):
    new = tour.copy()
    i, j = rng.choice(np.arange(1, len(tour)), size=2, replace=False)
    new[i], new[j] = new[j], new[i]
    return new


def reverse_move(tour, rng):
    new = tour.copy()
    i, j = np.sort(rng.choice(np.arange(1, len(tour)), size=2, replace=False))
    new[i:j + 1] = new[i:j + 1][::-1]
    return new


MOVES = {"swap": swap_move, "reverse": reverse_move}


def run_ba_tsp(D, seed, move="swap", settings=None, budget=DEFAULT_BUDGET):
    """Run BA on distance matrix D. Returns a dict of per-run results."""
    s = dict(DEFAULT_SETTINGS)
    if settings:
        s.update(settings)
    n_bees, m, e, nep, nsp = s["n"], s["m"], s["e"], s["nep"], s["nsp"]
    assert 0 < e <= m <= n_bees
    make_neighbour = MOVES[move]
    n_cities = D.shape[0]

    rng = np.random.default_rng(seed)

    def random_bee():
        return np.concatenate(([0], rng.permutation(np.arange(1, n_cities))))

    ev = Evaluator(lambda t: tour_length(t, D), budget)

    t0 = time.perf_counter()
    initial_best = None
    try:
        pop = [random_bee() for _ in range(n_bees)]
        fit = [ev(t) for t in pop]
        initial_best = ev.best_f

        while True:
            order = np.argsort(fit)
            sites = [(pop[i], fit[i]) for i in order[:m]]
            new_sites = []
            for rank, (t, ft) in enumerate(sites):
                k = nep if rank < e else nsp
                best_t, best_ft = t, ft
                for _ in range(k):
                    cand = make_neighbour(t, rng)
                    fc = ev(cand)              # counted even if rejected
                    if fc < best_ft:
                        best_t, best_ft = cand, fc
                new_sites.append((best_t, best_ft))
            scouts = []
            for _ in range(n_bees - m):
                t = random_bee()
                scouts.append((t, ev(t)))
            nxt = new_sites + scouts
            pop = [p[0] for p in nxt]
            fit = [p[1] for p in nxt]
    except BudgetExhausted:
        pass
    runtime = time.perf_counter() - t0

    assert validate_tour(ev.best_x, n_cities), "invalid tour returned"
    assert tour_length(ev.best_x, D) == ev.best_f
    return {
        "seed": seed,
        "move": move,
        "best_f": int(ev.best_f),
        "best_tour": ev.best_x,
        "initial_best": int(initial_best),
        "history": np.array(ev.history),
        "evals": ev.evals,
        "runtime": runtime,
        "settings": s,
    }


if __name__ == "__main__":
    # quick demo: one run per neighbourhood on each TSPLIB instance (seed 1)
    from pathlib import Path
    from tsp_loader import distance_matrix, load_tsplib

    data = Path(__file__).parent.parent / "data"
    for name, opt in OPTIMA.items():
        inst = load_tsplib(data / f"{name}.tsp")
        D = distance_matrix(inst["coords"])
        for mv in ("swap", "reverse"):
            r = run_ba_tsp(D, seed=1, move=mv)
            gap = 100 * (r["best_f"] - opt) / opt
            print(f"{name:9s} {mv:8s} initial={r['initial_best']:>7d}  "
                  f"final={r['best_f']:>7d}  gap={gap:6.2f}%  "
                  f"evals={r['evals']}  time={r['runtime']:.2f}s")
EOF