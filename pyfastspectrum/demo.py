#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyfastspectrum
import scipy.sparse.linalg as sla


SAMPLE_TYPES = {
    "poisson-disk": pyfastspectrum.SamplingType.Sample_Poisson_Disk,
    "farthest-point": pyfastspectrum.SamplingType.Sample_Farthest_Point,
    "random": pyfastspectrum.SamplingType.Sample_Random,
}

BACKENDS = {
    "auto": pyfastspectrum.SolverBackend.Auto,
    "cpu": pyfastspectrum.SolverBackend.CPU,
    "cuda": pyfastspectrum.SolverBackend.CUDA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pyfastspectrum eigenpairs against SciPy eigsh ground truth."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("suzanne.obj"),
        help="Path to an OBJ or OFF mesh file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of FastSpectrum samples / reduced basis size.",
    )
    parser.add_argument(
        "--sample-type",
        choices=tuple(SAMPLE_TYPES),
        default="farthest-point",
        help="Sampling strategy for FastSpectrum.",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(BACKENDS),
        default="cpu",
        help="FastSpectrum solver backend.",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=6,
        help="Number of non-zero modes to compare.",
    )
    parser.add_argument(
        "--eigsh-extra",
        type=int,
        default=4,
        help="Extra SciPy modes to compute beyond the requested non-zero modes.",
    )
    parser.add_argument(
        "--zero-eps",
        type=float,
        default=1e-8,
        help="Eigenvalues with absolute value <= zero-eps are treated as zero modes.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="Shift used by scipy.sparse.linalg.eigsh.",
    )
    parser.add_argument(
        "--which",
        default="LM",
        help="Selection mode passed to scipy.sparse.linalg.eigsh.",
    )
    return parser.parse_args()


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                polygon = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
                for index in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[index], polygon[index + 1]])

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def load_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        if first != "OFF":
            raise ValueError("OFF loader expected the first line to be OFF.")

        header = handle.readline().strip()
        while header.startswith("#") or not header:
            header = handle.readline().strip()
        num_vertices, num_faces, *_ = map(int, header.split())

        vertices = [list(map(float, handle.readline().split())) for _ in range(num_vertices)]
        faces: list[list[int]] = []
        for _ in range(num_faces):
            entries = list(map(int, handle.readline().split()))
            polygon = entries[1 : 1 + entries[0]]
            for index in range(1, len(polygon) - 1):
                faces.append([polygon[0], polygon[index], polygon[index + 1]])

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj(path)
    if suffix == ".off":
        return load_off(path)
    raise ValueError(f"Unsupported mesh extension: {path.suffix}")


def first_nonzero_index(values: np.ndarray, zero_eps: float) -> int:
    indices = np.flatnonzero(np.abs(values) > zero_eps)
    if indices.size == 0:
        raise ValueError("No non-zero eigenvalues were found.")
    return int(indices[0])


def main() -> None:
    args = parse_args()
    vertices, faces = load_mesh(args.model)

    fastspectrum = pyfastspectrum.FastSpectrum()
    basis, reduced_eigvecs, reduced_eigvals = fastspectrum.compute_eigenpairs(
        vertices,
        faces,
        args.num_samples,
        sample_type=SAMPLE_TYPES[args.sample_type],
        backend=BACKENDS[args.backend],
    )
    approx_eigvecs = basis @ reduced_eigvecs

    stiffness, mass = fastspectrum.assemble_laplacian(vertices, faces)
    scipy_k = min(
        vertices.shape[0] - 1,
        max(args.num_modes + args.eigsh_extra, args.num_modes + 1),
    )
    exact_eigvals, exact_eigvecs = sla.eigsh(
        stiffness,
        k=scipy_k,
        M=mass,
        sigma=args.sigma,
        which=args.which,
    )
    exact_order = np.argsort(exact_eigvals)
    exact_eigvals = exact_eigvals[exact_order]
    exact_eigvecs = exact_eigvecs[:, exact_order]

    approx_order = np.argsort(reduced_eigvals)
    approx_eigvals = np.asarray(reduced_eigvals)[approx_order]
    approx_eigvecs = np.asarray(approx_eigvecs)[:, approx_order]

    exact_start = first_nonzero_index(exact_eigvals, args.zero_eps)
    approx_start = first_nonzero_index(approx_eigvals, args.zero_eps)
    compare_count = min(
        args.num_modes,
        len(exact_eigvals) - exact_start,
        len(approx_eigvals) - approx_start,
    )
    if compare_count < 1:
        raise ValueError("Not enough non-zero modes are available for comparison.")

    exact_slice = slice(exact_start, exact_start + compare_count)
    approx_slice = slice(approx_start, approx_start + compare_count)
    exact_nonzero_vals = exact_eigvals[exact_slice]
    approx_nonzero_vals = approx_eigvals[approx_slice]
    exact_nonzero_vecs = exact_eigvecs[:, exact_slice]
    approx_nonzero_vecs = approx_eigvecs[:, approx_slice]

    rel_eig_errors = np.abs(approx_nonzero_vals - exact_nonzero_vals) / np.maximum(
        np.abs(exact_nonzero_vals), 1e-12
    )
    mass_approx = mass @ approx_nonzero_vecs
    vec_alignments = np.array(
        [
            abs(exact_nonzero_vecs[:, column].T @ mass_approx[:, column])
            for column in range(compare_count)
        ]
    )
    residuals = np.array(
        [
            np.linalg.norm(
                stiffness @ approx_nonzero_vecs[:, column]
                - approx_nonzero_vals[column] * (mass @ approx_nonzero_vecs[:, column])
            )
            / max(np.linalg.norm(stiffness @ approx_nonzero_vecs[:, column]), 1e-12)
            for column in range(compare_count)
        ]
    )

    print(f"Model: {args.model}")
    print(f"Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}")
    print(
        "FastSpectrum configuration:"
        f" samples={args.num_samples}, sample_type={args.sample_type}, backend={args.backend}"
    )
    print(
        "Zero modes:"
        f" exact={exact_start}, approx={approx_start}, compared_nonzero_modes={compare_count}"
    )
    print()
    print(
        f"{'mode':>4}  {'exact':>14}  {'approx':>14}  {'rel_error':>12}  {'|v^T M u|':>10}  {'residual':>10}"
    )
    for mode in range(compare_count):
        print(
            f"{mode + 1:>4}  "
            f"{exact_nonzero_vals[mode]:>14.8f}  "
            f"{approx_nonzero_vals[mode]:>14.8f}  "
            f"{rel_eig_errors[mode]:>12.6f}  "
            f"{vec_alignments[mode]:>10.6f}  "
            f"{residuals[mode]:>10.6f}"
        )

    print()
    print(
        "Summary:"
        f" mean_rel_error={rel_eig_errors.mean():.6f},"
        f" max_rel_error={rel_eig_errors.max():.6f},"
        f" mean_alignment={vec_alignments.mean():.6f},"
        f" min_alignment={vec_alignments.min():.6f},"
        f" mean_residual={residuals.mean():.6f},"
        f" max_residual={residuals.max():.6f}"
    )


if __name__ == "__main__":
    main()
