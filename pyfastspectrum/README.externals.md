# External dependencies for pyfastspectrum

`pyfastspectrum` expects two source trees under `ext/`, but those directories are ignored in the main G2PT repository so they do not get committed by accident.

Required layout:

```text
pyfastspectrum/
  ext/
    FastSpectrum/
    pybind11/
```

From `pyfastspectrum/`, run:

```bash
pwd  # should end with /pyfastspectrum
mkdir -p ext
git clone https://github.com/a-nasikun/FastSpectrum.git ext/FastSpectrum
git clone https://github.com/pybind/pybind11.git ext/pybind11
git -C ext/pybind11 checkout v3.0.2
```

Then build or install as usual, for example:

```bash
python3 -m pip install -e .
```

Recorded external revisions for this setup:

- `pybind11`: `v3.0.2` (from `ext/pybind11/include/pybind11/detail/common.h`)
- `FastSpectrum`: exact git tag is not recoverable from the current tree; use the same source snapshot as the existing local `ext/FastSpectrum/` checkout unless you intentionally re-vendor it

Notes:

- `CMakeLists.txt` reads FastSpectrum sources from `ext/FastSpectrum/Fast Spectrum` and pybind11 from `ext/pybind11`.
- FastSpectrum itself vendors additional third-party code inside its own tree; cloning the upstream repository is the intended setup.
- If you need fully reproducible setup for `FastSpectrum`, record the chosen commit or tag after cloning and keep it in this file.
