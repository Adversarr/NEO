# External dependencies for pyspec

`pyspec` expects header-only dependencies under `ext/`. That directory is ignored in the main G2PT repository so third-party sources stay local and do not get committed.

Required layout:

```text
pyspec/
  ext/
    pybind11/
    spectra/
```

From `pyspec/`, run:

```bash
pwd  # should end with /pyspec
mkdir -p ext
git clone https://github.com/pybind/pybind11.git ext/pybind11
git clone https://github.com/yixuan/spectra.git ext/spectra
git -C ext/pybind11 checkout v3.0.2
git -C ext/spectra checkout v1.2.0
```

Then install or build as usual:

```bash
python3 -m pip install -e .
```

Recorded external revisions for this setup:

- `pybind11`: `v3.0.2` (from `ext/pybind11/include/pybind11/detail/common.h`)
- `Spectra`: `v1.2.0` (from `ext/spectra/include/Spectra/Util/Version.h`)

Notes:

- `CMakeLists.txt` adds pybind11 from `ext/pybind11`.
- The binding includes Spectra headers from `ext/spectra/include`.
- Eigen3 is still expected from the system toolchain via `find_package(Eigen3 REQUIRED)`.
