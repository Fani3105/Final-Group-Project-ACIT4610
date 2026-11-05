import csv
import json

from pathlib import Path
from time import perf_counter

import numpy as np

from .portfolio import N_ASSETS, load_data, target
from .evolution_strategies import ALGS


# Output directory.
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# Experiment settings
BUDGET = 30000
SEEDS = list(range(10))

# Required return-target levels.
ALPHAS = {
    "025": 0.25,
    "050": 0.50,
    "075": 0.75
}


def cases():
    """Return all 15 experimental cases"""

    return [
        f"{ds}_alpha{key}"
        for ds in N_ASSETS
        for key in ALPHAS
    ]


def case_parts(case):
    """Extract dataset and alpha from a case name"""

    ds, key = case.split("_alpha")

    if ds not in N_ASSETS or key not in ALPHAS:
        raise ValueError(
            f"Unknown case: {case}"
        )

    return ds, ALPHAS[key]


def run_one(case, alg, seed, budget=BUDGET, data=None):
    """Run one algorithm on one experimental case"""

    ds, a = case_parts(case)

    # Load local dataset if not already provided.
    if data is None:
        r, cov = load_data(ds)
    else:
        r, cov = data

    # Calculate minimum required return
    min_r = target(r, a)

    # A reproducible random generator
    rng = np.random.default_rng(seed)

    # Measure execution time.
    start = perf_counter()

    # Execute selected ES algorithm
    out = ALGS[alg](
        r,
        cov,
        min_r,
        rng,
        budget
    )

    seconds = perf_counter() - start

    # Store results from this run.
    row = {
        "case": case,
        "alg": alg,
        "seed": seed,
        "n": len(r),
        "target": min_r,
        "variance": out["var"],
        "return": out["ret"],
        "valid": out["valid"],
        "runtime_s": seconds,
        "evals": out["evals"],
        "weights": json.dumps(out["w"].tolist())
    }

    return row, out["hist"]


def summary(rows):
    """Calculate summary statistics for each case and algorithm"""

    ans = []

    for case in cases():

        for alg in ALGS:

            # Find all runs for the selected case and algorithm.
            group = [
                x for x in rows
                if x["case"] == case
                and x["alg"] == alg
            ]

            if not group:
                continue

            # Use only feasible runs for variance statistics.
            vals = [
                float(x["variance"])
                for x in group
                if x["valid"]
            ]

            # Runtime includes every run.
            times = [
                float(x["runtime_s"])
                for x in group
            ]

            ans.append({
                "case": case,
                "alg": alg,
                "n": group[0]["n"],
                "target": group[0]["target"],

                "best_var": (
                    min(vals)
                    if vals else float("nan")
                ),

                "mean_var": (
                    float(np.mean(vals))
                    if vals else float("nan")
                ),

                "std_var": (
                    float(np.std(vals, ddof=1))
                    if len(vals) > 1
                    else float("nan")
                ),

                "valid_runs": len(vals),

                "feasible_pct": (
                    100 * len(vals) / len(group)
                ),

                "mean_runtime_s": float(
                    np.mean(times)
                )
            })

    return ans


def write_csv(path, rows):
    """Save a list of dictionaries as a CSV file"""

    if not rows:
        return

    with path.open("w", newline="") as f:

        wr = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        wr.writeheader()
        wr.writerows(rows)


def run_All(budget=BUDGET):
    """Run all datasets, return targets, algorithms and seeds"""

    OUT.mkdir(exist_ok=True)

    # Directory for convergence histories.
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    rows = []

    for ds in N_ASSETS:

        # Load each dataset only once.
        data = load_data(ds)

        for key in ALPHAS:

            case = f"{ds}_alpha{key}"

            for alg in ALGS:

                for seed in SEEDS:

                    # Execute one experiment.
                    row, hist = run_one(
                        case,
                        alg,
                        seed,
                        budget,
                        data=data
                    )

                    rows.append(row)

                    # Save convergence history.
                    path = logs / (
                        f"{case}_{alg}_seed{seed}.npz"
                    )

                    np.savez_compressed(
                        path,
                        hist=hist
                    )

                    # Print a short progress summary.
                    print(
                        f"{case:18} "
                        f"{alg:5} "
                        f"seed={seed} "
                        f"var={row['variance']:.8g}",
                        flush=True
                    )

                    # Save completed results after each run.
                    write_csv(
                        OUT / "runs.csv",
                        rows
                    )

    # Generate final statistics.
    write_csv(
        OUT / "summary.csv",
        summary(rows)
    )

    return rows