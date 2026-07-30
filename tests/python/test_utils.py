import numpy as np
from heapq import heappop, heappush
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from murty import ASSIGNMENT_EMPTY, ASSIGNMENT_UNMATCHED, Assignment


# ---------- UTILS ----------------

def add_dummy_cols(mat):
    rows, cols = mat.shape
    ret = np.full((rows, rows + cols), np.inf)
    ret[:, :cols] = mat
    np.fill_diagonal(ret[:, cols:], 0)
    return ret

def mat_from_assignment(assignment: Assignment, shape):
    mat = np.zeros(shape, dtype=int)
    for r, c in assignment.ass:
        assert r != ASSIGNMENT_UNMATCHED
        assert c != ASSIGNMENT_UNMATCHED
        if c != ASSIGNMENT_EMPTY and r != ASSIGNMENT_EMPTY:
            mat[r, c] = 1
    return mat

def cost_from_assignment(C, assignment: Assignment):
    cost = 0.
    for r, c in assignment.ass:
        assert r != ASSIGNMENT_UNMATCHED
        assert c != ASSIGNMENT_UNMATCHED
        if c != ASSIGNMENT_EMPTY and r != ASSIGNMENT_EMPTY:
            cost += C[r, c]
    return cost

# ---------- SOLVER ---------------

@dataclass
class QElem:
    C: np.ndarray
    assignment: np.ndarray
    cost: float
    def __lt__(self, other):
        return self.cost < other.cost
    
def ban_edge(mat, row, col):
    mat[row, col] = np.inf

def fix_edge(mat, row, col):
    val = mat[row, col]
    mat[:, col] = np.inf
    mat[row, :] = np.inf
    mat[row, col] = val

def split_by_assignment(mat, assignment):
    mat = mat.copy()
    ret = []
    for r, c in enumerate(assignment):
        cur_mat = mat.copy()
        ban_edge(cur_mat, r, c)
        ret.append(cur_mat)
        fix_edge(mat, r, c)
    return ret

def solve_ref_full(C):
    try:
        row_idx, col_idx = linear_sum_assignment(C)
    except ValueError:
        return None, np.inf
        
    assignment = np.full(C.shape[0], -1, dtype=int)
    for i, r in enumerate(row_idx):
        assignment[r] = col_idx[i]
        
    assert assignment[assignment == -1].sum() == 0
    cost = C[row_idx, col_idx].sum()
    return assignment, cost

def solve_ref(C, K):
    rows, cols = C.shape
    C_dummy = add_dummy_cols(C)

    assignments, costs, heap = [], [], []

    assignment, cost = solve_ref_full(C_dummy)
    if np.isinf(cost):
        return costs, assignments
    heappush(heap, QElem(C_dummy, assignment, cost))

    while len(assignments) < K and heap:
        elem = heappop(heap)
        assignments.append(elem.assignment)
        costs.append(elem.cost)
        
        if len(assignments) == K:
            break
            
        split_mats = split_by_assignment(elem.C, elem.assignment)
        for mat in split_mats:
            assignment, cost = solve_ref_full(mat)
            if not np.isinf(cost):
                heappush(heap, QElem(mat, assignment, cost))
    
    for ass in assignments:
        for i in range(rows):
            if ass[i] >= cols:
                ass[i] = -1

    return assignments, costs

def solve_ref_subsets(C, K, row_subs, col_subs, base_costs = None):
    if len(row_subs) == 0:
        row_subs = [np.arange(C.shape[0], dtype=int)]
    if len(col_subs) == 0:
        col_subs = [np.arange(C.shape[1], dtype=int)]

    n = max(len(row_subs), len(col_subs))
    assert len(row_subs) in (1, n), "Row subsets must be size 0, 1 or n"
    assert len(col_subs) in (1, n), "Col subsets must be size 0, 1 or n"

    sols = []

    for i in range(n):
        rows = row_subs[0] if len(row_subs) == 1 else row_subs[i]
        cols = col_subs[0] if len(col_subs) == 1 else col_subs[i]
        base = base_costs[i] if (base_costs is not None and len(base_costs) > 0) else 0

        mat = C[np.ix_(rows, cols)]
        tmp_asss, tmp_costs = solve_ref(mat, K)

        for ass, cost in zip(tmp_asss, tmp_costs):
            new_ass = np.full(C.shape[0], -1, dtype=int) 
            
            for row, col in enumerate(ass):
                if col >= 0:
                    new_ass[rows[row]] = cols[col]
                else:
                    new_ass[rows[row]] = -2
            sols.append((cost + base, new_ass))

    sols.sort(key=lambda x: x[0])
    sols = sols[:K]
    
    final_asss = np.array([x[1] for x in sols])
    final_costs = np.array([x[0] for x in sols])

    return final_asss, final_costs