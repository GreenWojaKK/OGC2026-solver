"""Run one instance through the active solver and verify the result."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "solver"


def _parse_env(rows):
    result = {}
    for row in rows:
        key, separator, value = row.partition("=")
        if not separator or not key:
            raise ValueError(f"expected KEY=VALUE, got {row!r}")
        result[key] = value
    return result


def execute(instance, timelimit, threads=4, env=None):
    instance = Path(instance).resolve()
    run_env = {str(key): str(value) for key, value in (env or {}).items()}
    os.environ.update(run_env)
    os.environ["NUMBA_NUM_THREADS"] = str(int(threads))
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ogc2026_numba")
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    problem = json.loads(instance.read_text())
    sys.path.insert(0, str(SOLVER))
    os.chdir(SOLVER)
    import_started = time.time()
    from myalgorithm import RUN_PROVENANCE, algorithm
    import_s = time.time() - import_started

    started = time.time()
    solution = algorithm(problem, timelimit=float(timelimit))
    wall_s = time.time() - started
    provenance = json.loads(json.dumps(RUN_PROVENANCE))
    result = {
        "instance": str(instance),
        "timelimit": float(timelimit),
        "status": "no_solution" if solution is None else "ok",
        "feasible": False,
        "objective": None,
        "obj1": None,
        "obj2": None,
        "obj3": None,
        "wall_s": round(wall_s, 3),
        "import_s": round(import_s, 3),
        "solution_hash": None,
        "provenance": provenance,
    }
    if solution is None:
        return result

    from utils import check_feasibility
    checked = check_feasibility(problem, solution)
    encoded = json.dumps(solution, sort_keys=True).encode()
    result.update(
        feasible=bool(checked.get("feasible")),
        objective=checked.get("objective"),
        obj1=checked.get("obj1"),
        obj2=checked.get("obj2"),
        obj3=checked.get("obj3"),
        solution_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )
    if provenance:
        provenance["verified_final_objective"] = checked.get("objective")
        provenance["verified_final_feasible"] = bool(
            checked.get("feasible"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path)
    parser.add_argument("--tl", type=float, default=180.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--env", action="append", default=[], metavar="K=V")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output.resolve() if args.output else None
    result = execute(
        args.instance, args.tl, threads=args.threads,
        env=_parse_env(args.env))
    encoded = json.dumps(result, indent=1)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")
    print(encoded)
    raise SystemExit(0 if result["feasible"] else 2)


if __name__ == "__main__":
    main()
