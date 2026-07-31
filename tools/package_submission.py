"""Create a flat submission archive from the active runtime files."""

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "solver"


def main():
    version = (ROOT / "VERSION").read_text().strip()
    destination = ROOT / "artifacts" / f"submission-{version}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [
        path for path in sorted(SOLVER.iterdir())
        if (
            path.suffix in (".py", ".so")
            and path.name != "rc_integration_check.py"
        )
    ]
    required = {"myalgorithm.py", "constructor.py", "utils.py"}
    if not required.issubset(path.name for path in files):
        raise RuntimeError("active runtime file set is incomplete")
    with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    print(destination)
    print(f"files={len(files)}")


if __name__ == "__main__":
    main()

