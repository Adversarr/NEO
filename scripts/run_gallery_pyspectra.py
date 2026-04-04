"""Non-destructive gallery launcher for pyspec experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp/pretrain/infer_pyspectra.py on existing gallery cases without overwriting results."
    )
    parser.add_argument(
        "--gallery_dir",
        type=str,
        default="tmp/base_sampling",
        help="Existing gallery root. Expected layout: <gallery_dir>/<mesh>/<case>/sample_points.npy",
    )
    parser.add_argument("--mesh_glob", type=str, default="*", help="Glob for mesh directories under gallery_dir.")
    parser.add_argument("--case_glob", type=str, default="*", help="Glob for case directories under each mesh dir.")
    parser.add_argument("--k", type=int, default=128, help="Number of eigenpairs to compute.")
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--maxit", type=int, default=1000)
    parser.add_argument("--ncv", type=int, default=0)
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force through to infer_pyspectra.py. By default existing results are never overwritten.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> bool:
    print(f"run: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"failed: {exc}")
        return False
    return True


def iter_mesh_dirs(gallery_dir: Path, mesh_glob: str) -> list[Path]:
    return sorted(path for path in gallery_dir.glob(mesh_glob) if path.is_dir())


def pending_case_names(mesh_dir: Path, case_glob: str, force: bool) -> list[str]:
    pending: list[str] = []
    for case_dir in sorted(path for path in mesh_dir.glob(case_glob) if path.is_dir()):
        if not (case_dir / "sample_points.npy").exists():
            continue
        if not force and (case_dir / "results_spectra_exp.json").exists():
            continue
        pending.append(case_dir.name)
    return pending


def main() -> int:
    args = parse_args()
    gallery_dir = Path(args.gallery_dir)
    if not gallery_dir.exists():
        print(f"gallery_dir does not exist: {gallery_dir}")
        return 1

    mesh_dirs = iter_mesh_dirs(gallery_dir, args.mesh_glob)
    if not mesh_dirs:
        print(f"no mesh directories matched in {gallery_dir}")
        return 1

    total_pending = 0
    total_skipped = 0
    total_failed = 0

    for mesh_dir in mesh_dirs:
        pending = pending_case_names(mesh_dir, args.case_glob, args.force)
        all_cases = [path for path in mesh_dir.glob(args.case_glob) if path.is_dir() and (path / "sample_points.npy").exists()]
        skipped = len(all_cases) - len(pending)
        total_pending += len(pending)
        total_skipped += skipped

        if not pending:
            print(f"[skip] {mesh_dir}: no pending cases")
            continue

        print(f"[mesh] {mesh_dir}: pending={len(pending)} skipped={skipped}")
        for case_name in pending:
            case_cmd = [
                args.python,
                "exp/pretrain/infer_pyspectra.py",
                "--data_dir",
                str(mesh_dir),
                "--glob",
                case_name,
                "--k",
                str(args.k),
                "--sigma",
                str(args.sigma),
                "--tol",
                str(args.tol),
                "--maxit",
                str(args.maxit),
            ]
            if args.ncv > 0:
                case_cmd.extend(["--ncv", str(args.ncv)])
            if args.force:
                case_cmd.append("--force")
            if args.dry_run:
                print(f"dry-run: {' '.join(case_cmd)}")
                continue
            if not run_command(case_cmd):
                total_failed += 1

    print(f"done: pending={total_pending} skipped={total_skipped} failed={total_failed}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
