# External dependencies for pypcdlaplace

`pypcdlaplace` builds against external source trees under `ext/`. Those directories are intentionally ignored in the main G2PT repository so vendored dependencies do not get added to commits.

Required layout:

```text
pypcdlaplace/
  ext/
    cgal/
    pybind11/
```

From `pypcdlaplace/`, run:

```bash
pwd  # should end with /pypcdlaplace
mkdir -p ext
git clone https://github.com/CGAL/cgal.git ext/cgal
git clone https://github.com/pybind/pybind11.git ext/pybind11
git -C ext/cgal checkout v6.2
git -C ext/pybind11 checkout v3.0.2
```

Then install or build as usual:

```bash
python3 -m pip install -e .
```

Recorded external revisions for this setup:

- `pybind11`: `v3.0.2` (from `ext/pybind11/include/pybind11/detail/common.h`)
- `CGAL`: `v6.2` is the recommended checkout; the current local tree reports `6.2-dev` headers, so re-check if your local clone differs

Notes:

- `CMakeLists.txt` resolves CGAL from `ext/cgal` with `find_package(CGAL CONFIG REQUIRED ...)`.
- `pybind11` is added with `add_subdirectory(ext/pybind11)`.
- You still need a working C++ toolchain plus BLAS/LAPACK development libraries on the system.
