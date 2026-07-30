"""
Typing hints for the C++ functions
"""

import numpy as np
from typing import Optional, List, Sequence, Union

from murty.workers import AssignmentWorkers, Assignment, MurtyWorkers, SparseMatrix


# --------------------- MURTY --------------------------------

def assignment(
    C: np.ndarray,
    workers: Optional[AssignmentWorkers] = None,
    max_cost: float = ...
) -> Assignment:
    """
    Solves the dense optimal assignment problem using the Successive Shortest Path (SSP) algorithm

    Parameters:
    - C: 2D numpy array representing the cost matrix
    - workers: Optional pre-allocated AssignmentWorkers for reusing memory
    - max_cost: Maximum enabled cost of the matching

    Returns:
    - The optimal Assignment
    """
    ...

def assignment_sparse(
    C: np.ndarray,
    max_per_row: int = ...,
    max_val: float = ...,
    workers: Optional[AssignmentWorkers] = None,
    max_cost: float = ...
) -> Assignment:
    """
    Solves the sparse optimal assignment problem
    Implicitly sparsifies the dense matrix based on given thresholds before solving

    Parameters:
    - C: 2D numpy array representing the base cost matrix
    - max_per_row: Maximum number of valid edges to keep per row (default: infinity)
    - max_val: Maximum cost threshold for an edge to be considered valid (default: infinity)
    - workers: Optional pre-allocated AssignmentWorkers for reusing memory
    - max_cost: Maximum enabled cost of the matching

    Returns:
    - The optimal Assignment
    """
    ...

def murty(
    C: Union[np.ndarray, SparseMatrix],
    K: int,
    row_subsets: Sequence[Sequence[int]] = ...,
    col_subsets: Sequence[Sequence[int]] = ...,
    base_costs: Sequence[float] = ...,
    workers: Optional[MurtyWorkers] = None,
    max_cost: float = ...
) -> List[Assignment]:
    """
    Solves for the K-best dense assignments using Murty's algorithm.
    Can simultaneously solve for multiple sub-matrices if subsets are provided

    Parameters:
    - C: The master cost matrix
    - K: The number of best assignments to find
    - row_subsets: List of row-index lists to form sub-matrices. If empty, uses all rows
    - col_subsets: List of column-index lists to form sub-matrices. If empty, uses all columns
    - base_costs: Base cost for each subset, default is 0
    - workers: Optional pre-allocated MurtyWorkers for reusing memory
    - max_cost: Maximum enabled cost of the matching

    Returns:
    - List of the k-best assignments
    """
    ...

def murty_sparse(
    C: np.ndarray,
    K: int,
    max_per_row: int = ...,
    max_val: float = ...,
    row_subsets: Sequence[Sequence[int]] = ...,
    col_subsets: Sequence[Sequence[int]] = ...,
    base_costs: Sequence[float] = ...,
    workers: Optional[MurtyWorkers] = None,
    max_cost: float = ...
) -> List[Assignment]:
    """
    Solves for the K-best sparse assignments using Murty's algorithm

    Parameters:
    - C: 2D numpy array representing the master cost matrix
    - K: The number of best assignments to find
    - max_per_row: Maximum number of valid edges to keep per row (default: infinity)
    - max_val: Maximum cost threshold for an edge to be considered valid (default: infinity)
    - row_subsets: List of row-index lists to form sub-matrices. If empty, uses all rows
    - col_subsets: List of column-index lists to form sub-matrices. If empty, uses all columns
    - base_costs: Base cost for each subset, default is 0
    - workers: Optional pre-allocated MurtyWorkers for reusing memory
    - max_cost: Maximum enabled cost of the matching

    Returns:
    - List of the k-best assignments
    """
    ...
