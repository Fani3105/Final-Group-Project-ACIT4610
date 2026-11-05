"""Sanity checks. Run from the project root:  python src/tests.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from functions import FUNCTIONS, ackley, rastrigin, rosenbrock, sphere
from tsp_loader import distance_matrix, load_tsplib, tour_length, validate_tour

DATA = Path(__file__).parent.parent / "data"


def test_functions():
    # 1) value at the known minimiser, D = 10 and 30
    for name, (f, lo, hi, xstar) in FUNCTIONS.items():
        for D in (10, 30):
            val = f(np.full(D, xstar))
            assert abs(val) < 1e-9, (name, D, val)
    # 2) one other point, values worked out by hand
    assert abs(sphere([1, 2]) - 5.0) < 1e-12
    assert abs(rosenbrock([0, 0]) - 1.0) < 1e-12
    assert abs(rastrigin([1, 1]) - 2.0) < 1e-9
    assert abs(ackley([1, 1]) - 3.6253849384) < 1e-6
    print("functions OK")


def test_distance_rounding():
    # 3-4-5 triangle: exact distances; also a case needing rounding
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.4], [0.0, 1.6]])
    D = distance_matrix(coords)
    assert D[0, 1] == 5
    assert D[0, 2] == 1   # 1.4 + 0.5 = 1.9 -> floor 1
    assert D[0, 3] == 2   # 1.6 + 0.5 = 2.1 -> floor 2
    assert (D == D.T).all() and (np.diag(D) == 0).all()
    # square with side 10: tour 0-1-2-3 has length 40
    sq = distance_matrix(np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float))
    assert tour_length([0, 1, 2, 3], sq) == 40
    assert validate_tour(np.array([0, 2, 1, 3]), 4)
    assert not validate_tour(np.array([0, 2, 2, 3]), 4)   # repeated city
    assert not validate_tour(np.array([1, 0, 2, 3]), 4)   # city 1 not first
    print("distance + tour checks OK")


def test_tsplib_files():
    expected = {"eil51": 51, "berlin52": 52, "st70": 70,
                "eil76": 76, "kroA100": 100, "ch150": 150}
    for name, n in expected.items():
        path = DATA / f"{name}.tsp"
        if not path.exists():
            print(f"  skipped {name} (file not found)")
            continue
        inst = load_tsplib(path)
        assert inst["dimension"] == n and inst["coords"].shape == (n, 2)
        D = distance_matrix(inst["coords"])
        if name == "berlin52":
            assert D[0, 1] == 666   # city 1 (565,575) to city 2 (25,185)
        print(f"  {name}: {n} cities loaded and validated")
    print("TSPLIB file checks OK")


if __name__ == "__main__":
    test_functions()
    test_distance_rounding()
    test_tsplib_files()