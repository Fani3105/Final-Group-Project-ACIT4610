import numpy as np

from .portfolio import repair, ret, variance, feasible


# Population sizes
MU = 10
LAM = 70

# Step-size limits
LOW = 1e-8
HIGH = 1.0


def result(w, v, hist, r, min_r):
    """Return the best solution found during the search"""

    return {
        "w": w.copy(),
        "var": v,
        "ret": ret(w, r),
        "valid": feasible(w, r, min_r),
        "evals": len(hist),
        "hist": hist
    }


def one_one(r, cov, min_r, rng, budget=30000):
    """(1+1)-ES with the 1/5th success rule"""

    if budget < 1:
        raise ValueError(
            "Budget must be positive"
        )

    n = len(r)

    # Generate and repair one initial parent
    w = repair(
        rng.uniform(0, 1, n),
        r,
        min_r
    )

    v = variance(w, cov)

    # Best-so-far archive
    best_w = w.copy()
    best_v = v

    # Initial mutation step size
    sig = 0.1 / np.sqrt(n)

    # Store convergence history
    hist = np.empty(budget)
    hist[0] = best_v

    # Each iteration evaluates one offspring
    for k in range(1, budget):

        # Gaussian mutation
        child = w + sig * rng.normal(size=n)

        # Repair the new solution
        child = repair(child, r, min_r)

        # Calculate portfolio variance
        val = variance(child, cov)

        # Check whether the offspring is better
        improved = val < v

        # Survivor selection
        if improved:
            w = child
            v = val

        # Update the best-so-far archive
        if val < best_v:
            best_w = child.copy()
            best_v = val

        # Adapt step size using the 1/5th success rule
        if improved:
            sig *= np.exp(1 / 3)
        else:
            sig *= np.exp(-1 / 12)

        # Keep step size within the required bounds
        sig = float(
            np.clip(sig, LOW, HIGH)
        )

        # Record the best variance found so far
        hist[k] = best_v

    return result(
        best_w,
        best_v,
        hist,
        r,
        min_r
    )


def _population(r, cov, min_r, rng, budget, plus):
    """Shared code for the two population-based ES variants"""

    if budget < MU:
        raise ValueError(
            f"Population ES needs at least {MU} evaluations"
        )

    n = len(r)

    # Self-adaptation parameters
    t0 = 1 / np.sqrt(2 * n)
    t1 = 1 / np.sqrt(2 * np.sqrt(n))

    # Convergence history
    hist = np.empty(budget)

    # Parent population
    parents = []

    # Best-so-far archive
    best_v = float("inf")
    best_w = None

    # Number of objective evaluations
    k = 0

    # Initialize the parent population
    for _ in range(MU):

        w = repair(
            rng.uniform(0, 1, n),
            r,
            min_r
        )

        # Each weight has its own step size
        sig = np.full(
            n,
            0.1 / np.sqrt(n)
        )

        v = variance(w, cov)

        # Store weights, step sizes and variance
        parents.append((w, sig, v))

        # Update best-so-far archive.
        if v < best_v:
            best_w = w.copy()
            best_v = v

        hist[k] = best_v
        k += 1

    # Evolutionary search
    while k < budget:

        children = []

        # Do not exceed 30,000 evaluations
        count = min(LAM, budget - k)

        for _ in range(count):

            # Randomly select two different parents
            i, j = rng.choice(
                MU,
                size=2,
                replace=False
            )

            w1, s1, _ = parents[i]
            w2, s2, _ = parents[j]

            # Arithmetic recombination of weights
            w = (w1 + w2) / 2

            # Geometric recombination of step sizes
            sig = np.sqrt(s1 * s2)

            # Self-adaptation
            z0 = rng.normal()
            zi = rng.normal(size=n)

            sig = sig * np.exp(
                t0 * z0 + t1 * zi
            )

            # Limit step sizes
            sig = np.clip(
                sig,
                LOW,
                HIGH
            )

            # Gaussian mutation
            w = w + sig * rng.normal(size=n)

            # Repair the offspring
            w = repair(w, r, min_r)

            # Evaluate portfolio variance
            v = variance(w, cov)

            # Store offspring
            children.append((w, sig, v))

            # Update best-so-far archive
            if v < best_v:
                best_w = w.copy()
                best_v = v

            # Record convergence
            hist[k] = best_v
            k += 1

        # Select the next parent population
        # Selection is unnecessary after the final budget
        if k < budget:

            if plus:

                # (mu+lambda)-ES:
                # Parents and offspring compete
                pool = parents + children

            else:

                # (mu,lambda)-ES:
                # Only offspring compete
                pool = children

            # Sort by portfolio variance
            # Lower variance is better
            pool.sort(
                key=lambda item: item[2]
            )

            # Keep the best 10 solutions
            parents = pool[:MU]

    return result(
        best_w,
        best_v,
        hist,
        r,
        min_r
    )


def plus_es(r, cov, min_r, rng, budget=30000):
    """(mu+lambda)-ES"""

    return _population(
        r,
        cov,
        min_r,
        rng,
        budget,
        plus=True
    )


def comma_es(r, cov, min_r, rng, budget=30000):
    """(mu,lambda)-ES"""

    return _population(
        r,
        cov,
        min_r,
        rng,
        budget,
        plus=False
    )


# Available algorithms
ALGS = {
    "one": one_one,
    "plus": plus_es,
    "comma": comma_es
}