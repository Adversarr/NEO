# pyspec

Minimal Python bindings for Spectra's symmetric generalized shift-invert eigensolver.

This project targets generalized eigenproblems of the form:

```text
A x = lambda B x
```

where:

- `A` is symmetric
- `B` is symmetric positive definite

The intended use case is computing the lowest generalized eigenpairs of sparse operators such as a mesh cotan Laplacian with a mass matrix.

## What It Provides

- A native pybind11 extension backed by Eigen + Spectra
- A small Python wrapper with a friendlier API
- A runnable demo using SciPy sparse matrices

## Layout

- `src/pyspec.cpp`: C++ binding implementation
- `pyspec/__init__.py`: Python wrapper API
- `script/demo.py`: small end-to-end example
- `ext/spectra`: vendored Spectra source
- `ext/pybind11`: vendored pybind11 source

## Build

Requirements:

- CMake >= 3.18
- a C++17 compiler
- Python 3
- Eigen3
- NumPy
- SciPy

Build the extension:

```bash
cmake -S . -B build
cmake --build build -j
```

This produces a native module in `build/pyspec/` named similar to:

```text
build/pyspec/_pyspec.cpython-312-x86_64-linux-gnu.so
```

## Install

For a normal Python package install:

```bash
pip install .
```

For editable development:

```bash
pip install -e .
```

## Python API

After building, you can use the Python wrapper from the repo root:

```python
import scipy.sparse as sp
from pyspec import solve_lowest_generalized_eigenpairs

A = sp.eye(10, format="csc")
B = sp.eye(10, format="csc")

result = solve_lowest_generalized_eigenpairs(A, B, 3)

print(result.status)
print(result.nconv)
print(result.eigenvalues)
print(result.eigenvectors.shape)
```

Primary entrypoints:

- `pyspec.solve_shift_invert_generalized(A, B, k, *, ncv=None, sigma=0.0, maxit=1000, tol=1e-10)`
- `pyspec.solve_lowest_generalized_eigenpairs(stiffness, mass, k, *, ncv=None, sigma=0.0, maxit=1000, tol=1e-10)`

The wrapper returns a `GeneralizedEigenResult` with:

- `eigenvalues`
- `eigenvectors`
- `nconv`
- `status`
- `converged`

## Input Expectations

- `A` and `B` should be SciPy sparse matrices
- they are converted internally to CSC format
- `A` and `B` must be square and have the same shape
- `B` must be positive definite
- shift-invert requires `A - sigma * B` to be factorizable

For lowest eigenpairs, a common choice is `sigma=0.0`.

## Demo

Run the example:

```bash
python3 script/demo.py
```

It constructs a simple sparse 1D Laplacian and solves for the lowest generalized eigenpairs.

## Notes

- The current binding is intentionally minimal.
- It currently targets `double` precision sparse inputs.
- The current solver path uses Spectra's `SymGEigsShiftSolver` with `GEigsMode::ShiftInvert`.
