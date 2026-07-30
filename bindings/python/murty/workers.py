import numpy as np
from numpy.typing import DTypeLike
from typing import Optional, Tuple, Type, Any
from . import _core


_DTYPE_SUFF = {
    "float64": "d",
    "float32": "f",
    np.dtype("float64"): "d",
    np.dtype("float32"): "f",
    np.float64: "d",
    np.float32: "f",
    float: "d",
}

def _get_binding_class(base_name: str, dtype: DTypeLike = np.float64, params: Optional[Tuple[Any, ...]] = None) -> Type:

    type_suff = _DTYPE_SUFF.get(dtype)
    if type_suff is None:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes are {_DTYPE_SUFF.keys()}")

    if params is not None:
        param_str = '_'.join([str(x) for x in params])
        class_name = f"{base_name}_{type_suff}_{param_str}"
        if hasattr(_core, class_name):
            return getattr(_core, class_name)
    
    class_name = f"{base_name}_{type_suff}"
    if hasattr(_core, class_name):
        return getattr(_core, class_name)
        
    raise NotImplementedError(f"No binding found for {base_name} with dtype={dtype} and params={'None' if params is None else params}")

def AssignmentWorkers(rows: int = 0, cols: int = 0, dtype: DTypeLike = np.float64):
    """
    Initializes AssignmentWorkers with the given dimensions and data type

    Parameters:
    - rows: Number of rows to initalize
    - cols: Number of columns to initalize
    - dtype: Data type
    """

    cls = _get_binding_class("_AssignmentWorkers", dtype)
    return cls(rows, cols)

def MurtyWorkers(K: int = 0, rows: int = 0, cols: int = 0, dtype: DTypeLike = np.float64):
    """
    Initializes MurtyWorkers with the given dimensions and data type

    Parameters:
    - K: Number of best solutions from murty
    - rows: Number of rows to initalize
    - cols: Number of columns to initalize
    - dtype: Data type
    """

    cls = _get_binding_class("_MurtyWorkers", dtype)
    return cls(K, rows, cols)

def Assignment(dtype: DTypeLike = np.float64):
    """
    Initializes Assignment
    """

    cls = _get_binding_class("_Assignment", dtype)
    return cls()

def SparseMatrix(arr: np.ndarray, max_per_row: int = 1000000000, max_val = np.inf, dtype: DTypeLike = np.float64):
    """
    Initializes sparse matrix

    Parameters:
    - arr: Matrix to be sparsified
    - rows: Maximum number of elements per row
    - cols: Maximum element value in the sparse matrix
    - dtype: Data type
    """

    cls = _get_binding_class("_SparseMatrix", dtype)
    return cls(arr, max_per_row, max_val)
