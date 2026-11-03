from pathlib import Path
import numpy as np


# Asset count in each dataset.
N_ASSETS = {
    "port1": 31,
    "port2": 85,
    "port3": 89,
    "port4": 98,
    "port5": 225
}


def load_data(name, folder=None):

    if folder is None:

        here = Path(__file__).resolve().parent

        paths = [
            here / "data" / f"{name}.txt",
            here.parent / "data" / f"{name}.txt"
        ]

        path = next(
            (p for p in paths if p.is_file()),
            paths[0]
        )

    else:
        path = Path(folder) / f"{name}.txt"

    # Read but ignore blank lines
    lines = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    # First line contains the number of assets.
    n = int(lines[0])

    if name in N_ASSETS and n != N_ASSETS[name]:
        raise ValueError(
            f"Invalid number of assets in {name}: expected {N_ASSETS[name]}, got {n}"
        )

    # Correct amout of lines in dataset: 1 asset count + n asset records + unique correlations
    exp = 1 + n + n * (n + 1) // 2

    if n < 2 or len(lines) != exp:
        raise ValueError(
            f"Invalid dataset: incorrect number of lines in {path}"
        )

    # Required returns and standard deviations
    r = np.zeros(n)
    s = np.zeros(n)

    for i in range(n):
        vals = lines[i + 1].split()

        if len(vals) != 2:
            raise ValueError(
                f"Expected 2 values for asset {i + 1}"
            )

        r[i] = float(vals[0])
        s[i] = float(vals[1])

    # Correlation matrix.
    c = np.zeros((n, n))
    seen = set()

    for line in lines[1 + n:]:

        i, j, val = line.split()

        # Change indices from starting at 1 to 0
        i = int(i) - 1
        j = int(j) - 1
        val = float(val)

        pair = (
            min(i, j),
            max(i, j)
        )

        if (
            not (0 <= i < n and 0 <= j < n)
            or pair in seen
        ):
            raise ValueError(
                f"Invalid or duplicate correlation record: {line}"
            )

        seen.add(pair)

        # The correlation matrix is symmetric.
        c[i, j] = val
        c[j, i] = val

    # Validate data values.
    if (
        not np.isfinite(r).all()
        or not np.isfinite(s).all()
        or not np.isfinite(c).all()
        or (s < 0).any()
        or (np.abs(c) > 1.00000001).any()
        or not np.allclose(np.diag(c), 1.0, atol=1e-8)
    ):
        raise ValueError(
            f"Invalid asset statistics in {path}"
        )

    # Covariance matrix
    cov = np.outer(s, s) * c

    # Assignment checkpoint for port1.
    if name == "port1":

        u = np.ones(n) / n

        ref = [
            0.003504064516129,
            0.001130937943724,
            0.007184532258065
        ]

        got = [
            float(r @ u),
            float(u @ cov @ u),
            float(0.5 * (r @ u + r.max()))
        ]

        if not np.allclose(
            got,
            ref,
            rtol=0,
            atol=1e-8
        ):
            raise ValueError(
                "port1 failed the assignment's numerical checkpoint"
            )

    return r, cov


def target(r, a):
    """Calculate the required minimum return."""

    return float(
        (1 - a) * r.mean() + a * r.max()
    )


def ret(w, r):
    """Calculate expected portfolio return."""

    return float(r @ w)


def variance(w, cov):
    """Calculate portfolio variance."""

    return float(w @ cov @ w)


def simplex(z):
    """Project weights onto the probability simplex."""

    # Sort weights from highest to lowest.
    v = np.sort(z)[::-1]

    total = 0.0
    t = 0.0

    for i, val in enumerate(v, start=1):
        total += val
        t_new = (total - 1) / i

        if val > t_new:
            t = t_new

    # Sum of all weights must be 1 and also weights can't be negative 
    return np.maximum(z - t, 0.0)


def repair(z, r, min_r):
    """Repair an invalid portfolio allocation."""

    # Step 1: Simplex projection.
    w = simplex(z)

    # Step 2: Check the return constraint.
    cur = ret(w, r)

    if cur < min_r:

        # Find the asset with the highest return.
        j = int(np.argmax(r))

        den = float(r[j] - cur)

        if den <= 0:
            raise ValueError(
                "Cannot reach the minimum required return"
            )

        # Calculate how much weight must be shifted toward the asset with the highest expected return
        b = (min_r - cur) / den

        # Move part of the allocation toward asset j.
        w = (1 - b) * w
        w[j] += b

    return w


def feasible(w, r, min_r):
    """Check the three portfolio constraints."""

    return (
        np.min(w) >= -1e-8
        and abs(float(w.sum()) - 1) <= 1e-8
        and ret(w, r) >= min_r - 1e-8
    )