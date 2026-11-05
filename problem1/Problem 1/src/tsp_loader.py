"""TSPLIB loading, validation, TSPLIB-rounded distances and tour utilities.

Tours are stored as NumPy arrays of 0-based city indices. City id 1 in the
file is index 0, and it is always the first entry of a tour. When a route
is displayed as a closed loop, append tour[0] at the end for display only.
"""
from pathlib import Path

import numpy as np


def load_tsplib(path):
    """Parse and validate a TSPLIB .tsp file (TYPE: TSP, EDGE_WEIGHT_TYPE: EUC_2D).

    Returns a dict with keys: name, dimension, ids (1..n), coords (n x 2 floats).
    Raises ValueError if any check fails.
    """
    path = Path(path)
    header = {}
    rows = []
    in_coords = False
    saw_coord_section = False

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "EOF":
                break
            if in_coords:
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"{path.name}: bad coordinate row: {line!r}")
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
            elif line.upper().startswith("NODE_COORD_SECTION"):
                in_coords = True
                saw_coord_section = True
            elif ":" in line:
                # handles both "KEY: value" and "KEY : value"
                key, val = line.split(":", 1)
                header[key.strip().upper()] = val.strip()

    # --- validation ---
    for key in ("NAME", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE"):
        if key not in header:
            raise ValueError(f"{path.name}: missing header field {key}")
    if not saw_coord_section:
        raise ValueError(f"{path.name}: missing NODE_COORD_SECTION")
    if header["TYPE"].upper() != "TSP":
        raise ValueError(f"{path.name}: TYPE must be TSP, got {header['TYPE']}")
    if header["EDGE_WEIGHT_TYPE"].upper() != "EUC_2D":
        raise ValueError(f"{path.name}: EDGE_WEIGHT_TYPE must be EUC_2D")
    n = int(header["DIMENSION"])
    if len(rows) != n:
        raise ValueError(f"{path.name}: DIMENSION is {n} but found {len(rows)} rows")
    ids = [r[0] for r in rows]
    if len(set(ids)) != n:
        raise ValueError(f"{path.name}: city identifiers are not unique")
    if sorted(ids) != list(range(1, n + 1)):
        raise ValueError(f"{path.name}: city identifiers must be 1..{n}")

    rows.sort(key=lambda r: r[0])
    coords = np.array([[r[1], r[2]] for r in rows], dtype=float)
    return {
        "name": header["NAME"],
        "dimension": n,
        "ids": np.arange(1, n + 1),
        "coords": coords,
    }


def distance_matrix(coords):
    """TSPLIB EUC_2D distances: d_ij = floor(sqrt(dx^2 + dy^2) + 0.5), as ints."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return np.floor(dist + 0.5).astype(np.int64)


def tour_length(tour, D):
    """Closed tour length: all consecutive legs plus the return leg."""
    tour = np.asarray(tour)
    return int(D[tour, np.roll(tour, -1)].sum())


def validate_tour(tour, n):
    """True if tour contains every city 0..n-1 exactly once and starts at city 0."""
    tour = np.asarray(tour)
    return (
        tour.shape == (n,)
        and tour[0] == 0
        and np.array_equal(np.sort(tour), np.arange(n))
    )