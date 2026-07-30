import pytest
import numpy as np
import murty
from .test_utils import solve_ref, cost_from_assignment, solve_ref_subsets

# (num_tests, max_size, max_k, min_val, max_val)
TEST_CONFIGS = [
    (100, 5, 500, -1e6, 1e6),   # Small matrices, high K
    (50, 20, 100, 0, 1e6),     # Medium, non-negative
    (50, 20, 100, -1e6, 0),    # Medium, non-positive
    (5, 50, 500, -1e6, 1e6),   # Large matrices
]

@pytest.mark.parametrize("num_tests, max_size, max_k, min_val, max_val", TEST_CONFIGS)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("use_worker", [False, True])
def test_murty_correctness(num_tests, max_size, max_k, min_val, max_val, sparse, use_worker):

    np.random.seed(42)
    worker = murty.MurtyWorkers(0, 0, 0) if use_worker else None
    
    for _ in range(num_tests):
        rows = np.random.randint(3, max_size + 1)
        cols = np.random.randint(3, max_size + 1)
        K = np.random.randint(1, max_k + 1)
        
        mat = np.random.uniform(min_val, max_val, (rows, cols))

        if sparse:
            asss_murty = murty.murty_sparse(mat, K, workers=worker)
        else:
            asss_murty = murty.murty(mat, K, workers=worker)

        _, costs_ref = solve_ref(mat, K)

        assert len(asss_murty) == len(costs_ref), f"Expected {len(costs_ref)} solutions, got {len(asss_murty)}"

        for i in range(len(costs_ref)):
            cost_murty = asss_murty[i].cost
            ass_murty = asss_murty[i]
            cost_ref = costs_ref[i]

            actual_cost = cost_from_assignment(mat, ass_murty)
            assert np.isclose(cost_murty, actual_cost, atol=1e-5), f"Solution {i}: Returned cost {cost_murty} != calculated {actual_cost}"
            assert np.isclose(cost_murty, cost_ref, atol=1e-5), f"Solution {i}: MTT Cost {cost_murty} != Ref Cost {cost_ref}\nMatrix:\n{mat}"

@pytest.mark.parametrize("num_tests, max_size, max_k, min_val, max_val", TEST_CONFIGS)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("use_worker", [False, True])
def test_murty_correctness_bans(num_tests, max_size, max_k, min_val, max_val, sparse, use_worker):

    np.random.seed(42)
    worker = murty.MurtyWorkers(0, 0, 0) if use_worker else None
    
    for _ in range(num_tests):
        rows = np.random.randint(3, max_size + 1)
        cols = np.random.randint(3, max_size + 1)
        K = np.random.randint(1, max_k + 1)
        
        mat = np.random.uniform(min_val, max_val, (rows, cols))
        bans = np.random.randint(0, rows * cols)
        ban_rows = np.random.randint(0, rows, bans)
        ban_cols = np.random.randint(0, cols, bans)
        mat[ban_rows, ban_cols] = np.inf

        mx = max(max_val*rows*2, 1e-5)

        if sparse:
            asss_murty = murty.murty_sparse(mat, K, workers=worker, max_val=max_val, max_cost=mx)
        else:
            asss_murty = murty.murty(mat, K, workers=worker, max_cost=mx)

        _, costs_ref = solve_ref(mat, K)

        assert len(asss_murty) == len(costs_ref), f"Expected {len(costs_ref)} solutions, got {len(asss_murty)}"

        for i in range(len(costs_ref)):
            cost_murty = asss_murty[i].cost
            ass_murty = asss_murty[i]
            cost_ref = costs_ref[i]

            actual_cost = cost_from_assignment(mat, ass_murty)
            assert np.isclose(cost_murty, actual_cost, atol=1e-5), f"Solution {i}: Returned cost {cost_murty} != calculated {actual_cost}"
            assert np.isclose(cost_murty, cost_ref, atol=1e-5), f"Solution {i}: MTT Cost {cost_murty} != Ref Cost {cost_ref}\nMatrix:\n{mat}"

@pytest.mark.parametrize("sparse", [False, True])
def test_murty_max_cost_limit(sparse):

    np.random.seed(42)
    K = 100
    worker = murty.MurtyWorkers(K, 10, 10)

    for _ in range(10):
        mat = np.random.uniform(10, 100, (6, 6))
        BEST = 30

        if sparse:
            base_sols = murty.murty_sparse(mat, K) 
        else:
            base_sols = murty.murty(mat, K)
        base_costs = [sol.cost for sol in base_sols]
        assert len(base_costs) == K, f"{K} solutions should be found"

        cost = base_costs[BEST]
        limit = cost + 1e-4

        if sparse:
            asss_murty = murty.murty_sparse(mat, K, max_cost=limit, workers=worker)
        else:
            asss_murty = murty.murty(mat, K, max_cost=limit, workers=worker)

        for i in range(len(asss_murty)):
            cost_murty = asss_murty[i].cost
            ass_murty = asss_murty[i]

            actual_cost = cost_from_assignment(mat, ass_murty)
            assert np.isclose(cost_murty, actual_cost, atol=1e-5), f"Solution {i}: Returned cost {cost_murty} != calculated {actual_cost}"
            assert cost_murty < limit + 1e-5, f"Solution {i}: Returned cost {cost_murty} is over limit of {limit}"

            if i < BEST:
                assert np.isclose(cost_murty, base_costs[i], atol=1e-5), "Solutions cost do not match"

@pytest.mark.parametrize("shape", [(100, 3), (100, 2), (100, 1), (1, 100), (2, 100), (3, 100)])
@pytest.mark.parametrize("sparse", [False, True])
def test_murty_rectangular(shape, sparse):

    np.random.seed(42)
    K = 30
    worker = murty.MurtyWorkers(K, *shape)

    for _ in range(10):
        mat = np.random.uniform(-100, 100, shape)

        if sparse:
            asss_murty = murty.murty_sparse(mat, K, workers=worker)
        else:
            asss_murty = murty.murty(mat, K, workers=worker)

        _ , costs_ref = solve_ref(mat, K)

        assert len(asss_murty) == len(costs_ref), f"Expected {len(costs_ref)} solutions, got {len(asss_murty)}"

        for i in range(len(asss_murty)):
            cost_murty = asss_murty[i].cost
            ass_murty = asss_murty[i]
            cost_ref = costs_ref[i]

            actual_cost = cost_from_assignment(mat, ass_murty)
            assert np.isclose(cost_murty, actual_cost, atol=1e-5), f"Solution {i}: Returned cost {cost_murty} != calculated {actual_cost}"
            assert np.isclose(cost_murty, cost_ref, atol=1e-5), f"Solution {i}: MTT Cost {cost_murty} != Ref Cost {cost_ref}\nMatrix:\n{mat}"

@pytest.mark.parametrize("val", [0., 5., -5.])
@pytest.mark.parametrize("sparse", [False, True])
def test_murty_ties(val, sparse):

    np.random.seed(42)
    K = 30
    worker = murty.MurtyWorkers(K, 25, 25)

    for _ in range(10):
        mat = np.full((25, 20), val)

        if sparse:
            asss_murty = murty.murty_sparse(mat, K, workers=worker)
        else:
            asss_murty = murty.murty(mat, K, workers=worker)

        _ , costs_ref = solve_ref(mat, K)

        for i in range(len(asss_murty)):
            cost_murty = asss_murty[i].cost
            ass_murty = asss_murty[i]
            cost_ref = costs_ref[i]

            actual_cost = cost_from_assignment(mat, ass_murty)
            assert np.isclose(cost_murty, actual_cost, atol=1e-5), f"Solution {i}: Returned cost {cost_murty} != calculated {actual_cost}"
            assert np.isclose(cost_murty, cost_ref, atol=1e-5), f"Solution {i}: MTT Cost {cost_murty} != Ref Cost {cost_ref}\nMatrix:\n{mat}"

@pytest.mark.parametrize("num_tests, max_size, max_k, min_val, max_val", TEST_CONFIGS)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("use_worker", [False, True])
@pytest.mark.parametrize("use_base_costs", [False, True])
def test_murty_subsets(num_tests, max_size, max_k, min_val, max_val, sparse, use_worker, use_base_costs):

    np.random.seed(42)
    worker = murty.MurtyWorkers(0, 0, 0) if use_worker else None

    for _ in range(num_tests):
        rows = np.random.randint(3, max_size + 1)
        cols = np.random.randint(3, max_size + 1)
        K = np.random.randint(1, max_k + 1)

        mat = np.random.uniform(min_val, max_val, (rows, cols))

        subs = np.random.randint(0, 11)
        row_subs = []
        col_subs = []
        
        if subs > 0:
            row_subs = [np.sort(np.random.choice(rows, np.random.randint(1, rows + 1), replace=False)).tolist() for _ in range(subs)]
            col_subs = [np.sort(np.random.choice(cols, np.random.randint(1, cols + 1), replace=False)).tolist() for _ in range(subs)]

            p = np.random.rand()
            if p < 0.2:
                row_subs = [row_subs[0]]
            elif p < 0.4:
                col_subs = [col_subs[0]]
            elif p < 0.6:
                row_subs = [row_subs[0]]
                col_subs = [col_subs[0]]

        n = max(len(row_subs), len(col_subs))
        base_costs = np.random.random(n) * 100. if use_base_costs else []

        if sparse:
            asss_murty = murty.murty_sparse(mat, K, row_subsets=row_subs, col_subsets=col_subs, base_costs=base_costs, workers=worker)
        else:
            asss_murty = murty.murty(mat, K, row_subsets=row_subs, col_subsets=col_subs, base_costs=base_costs, workers=worker)

        _, costs_ref = solve_ref_subsets(mat, K, row_subs, col_subs, base_costs)

        assert len(asss_murty) == len(costs_ref), f"Expected {len(costs_ref)} solutions, got {len(asss_murty)}"

        for i in range(len(costs_ref)):
            cost_murty = asss_murty[i].cost
            cost_ref = costs_ref[i]

            assert np.isclose(cost_murty, cost_ref, atol=1e-5), f"Solution {i}: MTT Cost {cost_murty} != Ref Cost {cost_ref}\nMatrix:\n{mat}"

@pytest.mark.parametrize("num_tests, max_size, max_k, min_val, max_val", TEST_CONFIGS)
@pytest.mark.parametrize("use_worker", [False, True])
def test_sparse(num_tests, max_size, max_k, min_val, max_val, use_worker):

    np.random.seed(42)
    worker = murty.MurtyWorkers(0, 0, 0) if use_worker else None

    for _ in range(num_tests):
        rows = np.random.randint(3, max_size + 1)
        cols = np.random.randint(3, max_size + 1)
        K = np.random.randint(1, max_k + 1)

        mat = np.random.uniform(min_val, max_val, (rows, cols))
        max_per_row = int(np.random.random() * cols)
        max_val_mat = np.random.random() * (max_val - min_val) + min_val
        smat = murty.SparseMatrix(mat, max_per_row, max_val_mat)

        asss_ref = murty.murty_sparse(mat, K, max_per_row=max_per_row, max_val=max_val_mat, workers=worker)
        asss_murty = murty.murty(smat, K, workers=worker)

        assert len(asss_murty) == len(asss_ref), f"Expected {len(asss_ref)} solutions, got {len(asss_murty)}"

        for i in range(len(asss_ref)):
            cost_murty = asss_murty[i].cost
            cost_ref = asss_ref[i].cost

            assert np.isclose(cost_murty, cost_ref, atol=1e-5), f"Solution {i}: MTT Cost {cost_murty} != Ref Cost {cost_ref}\nMatrix:\n{mat}"
            assert np.array_equal(asss_murty[i].ass, asss_ref[i].ass)