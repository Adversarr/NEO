from __future__ import annotations

import importlib
import importlib.util
import site
import sys
import sysconfig
from pathlib import Path

import numpy as np
from scipy import sparse


def _load_core_module():
    try:
        return importlib.import_module("pypcdlaplace._core")
    except ImportError as original_error:
        package_dir = Path(__file__).resolve().parent
        search_roots = [
            package_dir,
            package_dir.parent / "build" / "install" / "pypcdlaplace",
        ]

        for maybe_root in (
            sysconfig.get_path("platlib"),
            sysconfig.get_path("purelib"),
            site.getusersitepackages(),
        ):
            if maybe_root:
                search_roots.append(Path(maybe_root) / "pypcdlaplace")

        for maybe_root in site.getsitepackages():
            search_roots.append(Path(maybe_root) / "pypcdlaplace")

        seen = set()
        for root in search_roots:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            for candidate in sorted(root.glob("_core*.so")) + sorted(root.glob("_core*.pyd")):
                spec = importlib.util.spec_from_file_location("pypcdlaplace._core", candidate)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules["pypcdlaplace._core"] = module
                spec.loader.exec_module(module)
                return module

        raise original_error


_core = _load_core_module()


def pcdlp_matrix(
    points,
    tdim,
    *,
    nn: int = 10,
    hs: float = 2.0,
    rho: float = 3.0,
    htype: str = "ddr",
):
    point_array = np.asarray(points, dtype=np.float64)
    rows, cols, values, shape = _core.compute_triplets(
        point_array,
        int(tdim),
        nn=int(nn),
        hs=float(hs),
        rho=float(rho),
        htype=str(htype),
    )
    return sparse.coo_matrix((values, (rows, cols)), shape=shape, dtype=np.float64).tocsr()
