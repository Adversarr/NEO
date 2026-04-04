# Repository Guidelines

## Project Structure & Module Organization
The active project code lives in `pcdlaplace/`. Core C++ sources, headers, the demo entry point, and MATLAB wrappers are all under `pcdlaplace/src/` (for example, `point_cloud.cpp`, `lpmatrix.h`, `test_pcd.cpp`, and `pcdlp_matrix.m`). `pcdlaplace/README.md` gives a short project summary, and `pcdlaplace/src/README` contains the original algorithm and licensing notes. `ext/pybind11/` is a vendored Git submodule; treat it as third-party code and avoid editing it unless you are intentionally updating the dependency.

## Build, Test, and Development Commands
Initialize dependencies before building:

```bash
git submodule update --init --recursive
cmake -S pcdlaplace/src -B build/pcdlaplace
cmake --build build/pcdlaplace
```

This configures and builds the `demo` executable declared in `pcdlaplace/src/CMakeLists.txt`. Run the CLI smoke test with a local point-cloud file:

```bash
./build/pcdlaplace/demo path/to/cloud.pcd
```

For MATLAB/MEX workflows, use the legacy build script from `pcdlaplace/src`:

```bash
matlab -batch "cd('pcdlaplace/src'); build"
```

## Coding Style & Naming Conventions
Match the existing C++ style in `pcdlaplace/src`: same-line braces, compact whitespace, and minimal modern C++ features. Use `CamelCase` for types (`PCloud`, `VPCloud`), lowercase or snake_case for filenames (`point_cloud.cpp`), and keep function names descriptive (`generate_pcdlaplace_matrix_sparse_matlab`). Preserve header/source pairings and keep MATLAB wrapper names aligned with the compiled MEX entry points.

## Testing Guidelines
There is no dedicated unit-test framework yet. Validate changes by rebuilding `demo` and running it on a representative `.pcd` input; confirm it completes and writes the expected `out.txt`. For MATLAB-facing changes, rebuild the MEX target and exercise `pcdlp_matrix.m`. If you add new verification data, keep it small and document how to run it in the PR.

## Commit & Pull Request Guidelines
`git log` is currently empty, so no repository-specific commit convention exists yet. Use short, imperative commit subjects such as `Add bounds check for empty point clouds`. Keep commits focused. PRs should state the problem, summarize the approach, list the commands you ran, and note any dependency assumptions (for example CGAL, LAPACK, or MATLAB availability). Include sample input/output when behavior changes.
