import numpy as np
from numpy.typing import DTypeLike


class AssignmentWorkers:
    """
    Worker structure to prevent memory reallocation during assignment function
    """

    def __init__(self, rows: int = 0, cols: int = 0, dtype: DTypeLike = np.float64) -> None:
        """
        Initializes the AssignmentWorkers

        Parameters:
        - rows: Maximum number of rows
        - cols: Maximum number of columns
        - dtype: Data type
        """
        ...

    def reserve(self, rows: int, cols: int) -> None:
        """
        Reserves memory for future calls
        
        Parameters:
        - rows: Maximum number of rows
        - cols: Maximum number of columns
        """
        ...

class MurtyWorkers:
    """
    Worker structure to prevent memory reallocation during murty function
    """

    def __init__(self, K: int = 0, rows: int = 0, cols: int = 0, dtype: DTypeLike = np.float64) -> None:
        """
        Initializes the class MurtyWorkers

        Parameters:
        - K: Maximum number of best solution wanted from Murty
        - rows: Maximum number of rows
        - cols: Maximum number of columns
        - dtype: Data type
        """
        ...

    def reserve(self, K: int, rows: int, cols: int) -> None:
        """
        Reserves memory for future calls
        
        Parameters:
        - K: Maximum number of best solution wanted from Murty
        - rows: Maximum number of rows
        - cols: Maximum number of columns
        """
        ...

class Assignment:
    """
    Class containing matrix assignnent
    """

    def __init__(self) -> None:
        """
        Initializes the Assignment class
        """
        ...

    @property
    def ass(self) -> np.ndarray:
        """
        Returns:
        - np.ndarray of shape (N, 2) - the assignment pairs (row, col)
        """
        ...

    @property
    def cost(self) -> ...:
        """
        Returns:
        - The cost of the assignment
        """
        ...

class SparseMatrix:
    """
    SparseMatrix for Murty's algorithm
    """

    def __init__(arr: np.ndarray, max_per_row: int = 1000000000, max_val = np.inf, dtype: DTypeLike = np.float64) -> None:
        """
        Initializes sparse matrix

        Parameters:
        - arr: Matrix to be sparsified
        - rows: Maximum number of elements per row
        - cols: Maximum element value in the sparse matrix
        - dtype: Data type
        """
        ...
