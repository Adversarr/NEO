from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def _load_native_module() -> ModuleType:
    try:
        return import_module("._pyspec", __name__)
    except ModuleNotFoundError:
        try:
            return import_module("_pyspec")
        except ModuleNotFoundError:
            pass

        root = Path(__file__).resolve().parents[1]
        matches = sorted((root / "build" / "pyspec").glob("_pyspec*.so"))
        if not matches:
            raise ModuleNotFoundError(
                "Could not import '_pyspec'. Build the extension first with "
                "'cmake -S . -B build && cmake --build build'."
            )

        spec = spec_from_file_location("_pyspec", matches[0])
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load native module from {matches[0]}")

        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_native = _load_native_module()

openmp_enabled = bool(_native.openmp_enabled)
configured_num_threads = int(_native.configured_num_threads)


@dataclass(frozen=True)
class GeneralizedEigenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    nconv: int
    status: str

    @property
    def converged(self) -> bool:
        return self.status == "Successful"


def solve_shift_invert_generalized(
    A: Any,
    B: Any,
    k: int,
    *,
    ncv: int | None = None,
    sigma: float = 0.0,
    maxit: int = 1000,
    tol: float = 1e-10,
) -> GeneralizedEigenResult:
    raw = _native.solve_sym_shift_invert_generalized(
        A,
        B,
        k,
        -1 if ncv is None else ncv,
        sigma,
        maxit,
        tol,
    )
    return GeneralizedEigenResult(
        eigenvalues=np.asarray(raw["eigenvalues"]),
        eigenvectors=np.asarray(raw["eigenvectors"]),
        nconv=int(raw["nconv"]),
        status=str(raw["status"]),
    )


def set_num_threads(num_threads: int) -> None:
    _native.set_num_threads(int(num_threads))


def get_num_threads() -> int:
    return int(_native.get_num_threads())


def solve_lowest_generalized_eigenpairs(
    stiffness: Any,
    mass: Any,
    k: int,
    *,
    ncv: int | None = None,
    sigma: float = 0.0,
    maxit: int = 1000,
    tol: float = 1e-10,
) -> GeneralizedEigenResult:
    return solve_shift_invert_generalized(
        stiffness,
        mass,
        k,
        ncv=ncv,
        sigma=sigma,
        maxit=maxit,
        tol=tol,
    )


__all__ = [
    "GeneralizedEigenResult",
    "configured_num_threads",
    "get_num_threads",
    "openmp_enabled",
    "set_num_threads",
    "solve_shift_invert_generalized",
    "solve_lowest_generalized_eigenpairs",
]
