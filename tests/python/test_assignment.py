import pytest
import numpy as np
import murty
from scipy.optimize import linear_sum_assignment
from .test_utils import add_dummy_cols, cost_from_assignment

# (num_tests, max_size, min_val, max_val)
TEST_CONFIGS = [
    (1000, 10, -1e6, 1e6),   # Small dense
    (200, 50, -1e6, 1e6),    # Medium dense
    (200, 50, 0, 1e6),       # Non-negative
    (200, 50, -1e6, 0),      # Non-positive
]

@pytest.mark.parametrize("num_tests, max_size, min_val, max_val", TEST_CONFIGS)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("use_worker", [False, True])
def test_assignment_correctness(num_tests, max_size, min_val, max_val, sparse, use_worker):

    np.random.seed(42)
    
    for _ in range(num_tests):
        rows = np.random.randint(3, max_size + 1)
        cols = np.random.randint(3, max_size + 1)
        
        mat = np.random.uniform(min_val, max_val, (rows, cols))
        dummy_mat = add_dummy_cols(mat)

        worker = murty.AssignmentWorkers(rows, cols) if use_worker else None

        if sparse:
            ass_murty = murty.assignment_sparse(mat, workers=worker)
        else:
            ass_murty = murty.assignment(mat, workers=worker)

        actual_cost = cost_from_assignment(mat, ass_murty)
        assert np.isclose(ass_murty.cost, actual_cost, atol=1e-5), f"Returned cost {ass_murty.cost} != calculated cost {actual_cost}"

        row_idx, col_idx = linear_sum_assignment(dummy_mat)
        cost_optimal = dummy_mat[row_idx, col_idx].sum()
        assert np.isclose(ass_murty.cost, cost_optimal, atol=1e-5), f"MTT Cost {ass_murty.cost} != Scipy Cost {cost_optimal}\nMatrix:\n{mat}"
            

@pytest.mark.parametrize("num_tests, max_size, min_val, max_val", TEST_CONFIGS)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("use_worker", [False, True])
def test_assignment_correctness_bans(num_tests, max_size, min_val, max_val, sparse, use_worker):

    np.random.seed(42)
    
    for _ in range(num_tests):
        rows = np.random.randint(3, max_size + 1)
        cols = np.random.randint(3, max_size + 1)
        
        mat = np.random.uniform(min_val, max_val, (rows, cols))
        bans = np.random.randint(0, rows * cols)
        ban_rows = np.random.randint(0, rows, bans)
        ban_cols = np.random.randint(0, cols, bans)
        mat[ban_rows, ban_cols] = np.inf

        dummy_mat = add_dummy_cols(mat)
        worker = murty.AssignmentWorkers(rows, cols) if use_worker else None

        if sparse:
            # should sparsify the infinities
            ass_murty = murty.assignment_sparse(mat, workers=worker, max_val=max_val, max_cost=max_val*rows*2)
        else:
            ass_murty = murty.assignment(mat, workers=worker, max_cost=max_val*rows*2)

        actual_cost = cost_from_assignment(mat, ass_murty)
        assert np.isclose(ass_murty.cost, actual_cost, atol=1e-5), f"Returned cost {ass_murty.cost} != calculated cost {actual_cost}"

        row_idx, col_idx = linear_sum_assignment(dummy_mat)
        cost_optimal = dummy_mat[row_idx, col_idx].sum()
        assert np.isclose(ass_murty.cost, cost_optimal, atol=1e-5), f"MTT Cost {ass_murty.cost} != Scipy Cost {cost_optimal}\nMatrix:\n{mat}"

@pytest.mark.parametrize("sparse", [False, True])
def test_assignment_max_cost(sparse):

    np.random.seed(42)

    for _ in range(10):
        mat = np.random.uniform(10, 100, (10, 10)) 
        dummy_mat = add_dummy_cols(mat)
        
        row_idx, col_idx = linear_sum_assignment(dummy_mat)
        cost_optimal = dummy_mat[row_idx, col_idx].sum()
        limit = cost_optimal - 1.0
        
        if sparse:
            ass_murty = murty.assignment_sparse(mat, max_cost=limit)
        else:
            ass_murty = murty.assignment(mat, max_cost=limit)
            
        assert ass_murty.ass.size == 0, f"Expected Assignment array to be None when max_cost is exceeded, got {ass_murty.ass}"

@pytest.mark.parametrize("shape", [(100, 3), (100, 2), (100, 1), (1, 100), (2, 100), (3, 100)])
@pytest.mark.parametrize("sparse", [False, True])
def test_assignment_rectangular(shape, sparse):

    np.random.seed(42)

    for _ in range(10):
        mat = np.random.uniform(-100, 100, shape)
        dummy_mat = add_dummy_cols(mat)

        if sparse:
            ass_murty = murty.assignment_sparse(mat)
        else:
            ass_murty = murty.assignment(mat)

        actual_cost = cost_from_assignment(mat, ass_murty)
        assert np.isclose(ass_murty.cost, actual_cost, atol=1e-5), f"Returned cost {ass_murty.cost} != calculated cost {actual_cost}"

        row_idx, col_idx = linear_sum_assignment(dummy_mat)
        cost_optimal = dummy_mat[row_idx, col_idx].sum()
        assert np.isclose(ass_murty.cost, cost_optimal, atol=1e-5), f"Rectangular failure {shape}. MTT Cost {ass_murty.cost} != Scipy Cost {cost_optimal}\nMatrix:\n{mat}"

@pytest.mark.parametrize("val", [0., 5., -5.])
@pytest.mark.parametrize("sparse", [False, True])
def test_assignment_ties(val, sparse):

    np.random.seed(42)

    for _ in range(10):
        mat = np.full((25, 20), val)
        dummy_mat = add_dummy_cols(mat)

        if sparse:
            ass_murty = murty.assignment_sparse(mat)
        else:
            ass_murty = murty.assignment(mat)

        actual_cost = cost_from_assignment(mat, ass_murty)
        assert np.isclose(ass_murty.cost, actual_cost, atol=1e-5), f"Returned cost {ass_murty.cost} != calculated cost {actual_cost}"

        row_idx, col_idx = linear_sum_assignment(dummy_mat)
        cost_optimal = dummy_mat[row_idx, col_idx].sum()
        assert np.isclose(ass_murty.cost, cost_optimal, atol=1e-5), f"MTT Cost {ass_murty.cost} != Scipy Cost {cost_optimal}\nMatrix:\n{mat}"

def test_assignment_explicit_sparsification():

    np.random.seed(42)

    for _ in range(10):
        mat = np.random.uniform(10, 100, (25, 20))
        
        ass_murty = murty.assignment_sparse(mat, max_per_row=5)
        
        actual_cost = cost_from_assignment(mat, ass_murty)
        assert np.isclose(ass_murty.cost, actual_cost, atol=1e-5), f"Returned cost {ass_murty.cost} != calculated cost {actual_cost}"