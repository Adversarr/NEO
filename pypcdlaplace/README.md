# pypcdlaplace

`pypcdlaplace` binds the legacy point-cloud to Laplace matrix implementation in `pcdlaplace/src/` through `pybind11`, using the vendored `ext/pybind11` and `ext/cgal` trees.

## Build and install

```bash
python3 -m pip install .
```

The extension is built with `scikit-build-core` and CMake. It expects a working C++17 toolchain plus BLAS/LAPACK development libraries at build time.

If your environment still ships an older `pip` (for example Ubuntu's `pip 22.x`), use:

```bash
python3 -m pip install . --no-build-isolation
```

For local development from this repository, prefer an editable install so the Python package and compiled extension stay importable together:

```bash
python3 -m pip install -e .
```

## Python API

```python
import numpy as np
from pypcdlaplace import pcdlp_matrix

points = np.random.default_rng(0).normal(size=(128, 3))
points /= np.linalg.norm(points, axis=1, keepdims=True)

laplace = pcdlp_matrix(points, 2, nn=12, hs=2.0, rho=3.0, htype="ddr")
print(laplace.shape, laplace.nnz)
```

`pcdlp_matrix(points, tdim, *, nn=10, hs=2.0, rho=3.0, htype="ddr")` accepts a NumPy array of shape `(n_points, ambient_dim)` and returns a SciPy `csr_matrix`.

Supported `htype` values:

- `ddr`: compute `h` from the average neighborhood size times `hs`
- `psp`: use `hs` directly as `h`

## Demo

Run the demo module after installation:

```bash
python3 -m pypcdlaplace.demo
```

The demo generates deterministic points on the unit sphere, builds the Laplace matrix, reports matrix statistics, and prints a few smallest eigenvalues of the symmetrized operator.

## Tests

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests
```

## Notes

- The binding only covers the point-cloud Laplace matrix path from the original MATLAB wrapper `pcdlaplace/src/pcdlp_matrix.m`.
- The old MATLAB/MEX entrypoints remain in the tree, but this package does not build or expose them.
