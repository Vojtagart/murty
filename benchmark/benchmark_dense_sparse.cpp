#include <iostream>
#include <iomanip>
#include <random>
#include "benchmark_helpers.hpp"

using namespace murty::benchmark;

constexpr size_t SPARSITY = 20;

struct TestCase {
    size_t rows = 0, cols = 0, K = 0, num = 0;
};

std::vector<TestCase> tcs_single = {
    TestCase{3, 3, 1, 1000},
    TestCase{10, 10, 1, 1000},
    TestCase{25, 25, 1, 1000},
    TestCase{50, 50, 1, 500},
    TestCase{100, 100, 1, 100},
    TestCase{200, 200, 1, 100},
    TestCase{300, 300, 1, 50},
    TestCase{500, 500, 1, 25}
};

std::vector<TestCase> tcs_k = {
    TestCase{10, 10, 10, 100},
    TestCase{10, 10, 30, 100},
    TestCase{10, 10, 90, 100},
    TestCase{10, 10, 270, 100},

    TestCase{25, 25, 25, 100},
    TestCase{25, 25, 75, 100},
    TestCase{25, 25, 225, 100},
    TestCase{25, 25, 675, 100},

    TestCase{50, 50, 50, 40},
    TestCase{50, 50, 150, 40},
    TestCase{50, 50, 450, 40},
    TestCase{50, 50, 1350, 40},

    TestCase{100, 100, 100, 20},
    TestCase{100, 100, 300, 20},
    TestCase{100, 100, 900, 20},
    TestCase{100, 100, 2700, 20},

    TestCase{200, 200, 200, 10},
    TestCase{200, 200, 600, 10},
    TestCase{200, 200, 1800, 10},
    TestCase{200, 200, 5400, 10},
};

DMatrix get_random_matrix(size_t rows, size_t cols, double mn, double mx, std::mt19937& rng) {
    auto dist = std::uniform_real_distribution<double>(mn, mx);
    DMatrix ret(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        ret.data()[i] = dist(rng);
    }
    return ret;
}

int main() {
    CSVWriter csv("benchmark_dense_sparse.csv");
    murty::MurtyWorkers<double, int> W;

    std::cout << "---------------------------------------------\n";
    std::cout << " single best assignment\n";
    std::cout << "---------------------------------------------\n";

    for (auto tc : tcs_single) {
        std::cout << "[ROWS = " << tc.rows << ", COLS = " << tc.cols << "]\n";
        double t_dense = 0., t_sparse = 0., t_sparse_mat = 0.;
        std::mt19937 rng(42);
        for (size_t i = 0; i < tc.num; i++) {
            DMatrix C = get_random_matrix(tc.rows, tc.cols, -1, 1, rng);
            SMatrix SC(C, SPARSITY);
            auto r_dense = run_murty_single(C, W);
            auto r_sparse = run_murty_single(SC, W);
            t_dense += r_dense.time_us;
            t_sparse += r_sparse.time_us;
            auto start_sc = std::chrono::high_resolution_clock::now();
            SC.fill_from(C, SPARSITY);
            auto end_sc = std::chrono::high_resolution_clock::now();
            t_sparse_mat += std::chrono::duration<double, std::micro>(end_sc - start_sc).count();
        }
        t_dense /= tc.num;
        t_sparse /= tc.num;
        t_sparse_mat /= tc.num;
        std::cout << "Dense        : " << t_dense << " us\n";
        std::cout << "Sparse       : " << t_sparse << " us\n";
        std::cout << "Sparsifying  : " << t_sparse_mat << " us\n\n";
        csv.write_row("single", tc.rows, tc.cols, tc.K, tc.num, "Dense", t_dense);
        csv.write_row("single", tc.rows, tc.cols, tc.K, tc.num, "Sparse", t_sparse);
        csv.write_row("single", tc.rows, tc.cols, tc.K, tc.num, "Sparsifying", t_sparse_mat);
    }

    std::cout << "---------------------------------------------\n";
    std::cout << " K-best assigments\n";
    std::cout << "---------------------------------------------\n";

    for (auto tc : tcs_k) {
        std::cout << "[ROWS = " << tc.rows << ", COLS = " << tc.cols << ", K = " << tc.K << "]\n";
        double t_dense = 0., t_sparse = 0., t_sparse_mat = 0.;
        std::mt19937 rng(42);
        for (size_t i = 0; i < tc.num; i++) {
            DMatrix C = get_random_matrix(tc.rows, tc.cols, -1, 1, rng);
            SMatrix SC(C, SPARSITY);
            auto r_dense = run_murty_k(C, tc.K, W);
            auto r_sparse = run_murty_k(C, tc.K, W);
            t_dense += r_dense.time_us;
            t_sparse += r_sparse.time_us;
            auto start_sc = std::chrono::high_resolution_clock::now();
            SC.fill_from(C, SPARSITY);
            auto end_sc = std::chrono::high_resolution_clock::now();
            t_sparse_mat += std::chrono::duration<double, std::micro>(end_sc - start_sc).count();
        }
        t_dense /= tc.num;
        t_sparse /= tc.num;
        t_sparse_mat /= tc.num;
        std::cout << "Dense        : " << t_dense << " us\n";
        std::cout << "Sparse       : " << t_sparse << " us\n";
        std::cout << "Sparsifying  : " << t_sparse_mat << " us\n\n";
        csv.write_row("k_best", tc.rows, tc.cols, tc.K, tc.num, "Dense", t_dense);
        csv.write_row("k_best", tc.rows, tc.cols, tc.K, tc.num, "Sparse", t_sparse);
        csv.write_row("k_best", tc.rows, tc.cols, tc.K, tc.num, "Sparsifying", t_sparse_mat);
    }
}