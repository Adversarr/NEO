from typing import Any
import torch
import numpy as np
import scipy.sparse as sp

def ensure_numpy(x: Any) -> np.ndarray:
    """Convert inputs to a NumPy ndarray.

    Rules:
    - torch.Tensor -> detach to CPU and return `.numpy()`
    - scipy.sparse matrix -> convert via `.toarray()` (dense) and wrap with `np.asarray`
    - anything else -> `np.asarray(x)`
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    # Use issparse to accept any scipy.sparse subtype without breaking static analysis
    try:
        if sp.issparse(x):  # type: ignore[arg-type]
            # x is sparse; toarray() exists at runtime on scipy.sparse matrices
            return np.asarray(x.toarray())  # type: ignore[attr-defined]
    except Exception:
        pass
    return np.asarray(x)

def roundup(x: int | float, base: int = 16) -> int:
    """
    Round up the input integer to the nearest multiple of the specified base.

    Args:
        x (int | float): The input integer to round up.
        base (int): The base to which the input should be rounded up. Default is 16.

    Returns:
        int: The rounded-up integer.
    """
    x = int(x)  # Ensure x is an integer
    return (x + base - 1) // base * base if x % base != 0 else x

def roundup_16(x: int | float) -> int:
    """
    Round up the input integer to the nearest multiple of 16.

    Args:
        x (int | float): The input integer to round up.

    Returns:
        int: The rounded-up integer.
    """
    return roundup(x, 16)

def roundup_256(x: int | float) -> int:
    """
    Round up the input integer to the nearest multiple of 256.

    Args:
        x (int | float): The input integer to round up.

    Returns:
        int: The rounded-up integer.
    """
    return roundup(x, 256)
