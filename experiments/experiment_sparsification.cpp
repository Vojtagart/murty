#include "murty/solve.hpp"
#include "murty/sparse_matrix.hpp"
#include "murty/matrix.hpp"

#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

struct TestCase {
    size_t N, K = 0, num = 0;
};

std::vector<TestCase> tcs = {
    TestCase{10, 10, 300},
    TestCase{10, 30, 300},
    TestCase{10, 90, 300},
    TestCase{10, 270, 300},

    TestCase{25, 25, 200},
    TestCase{25, 75, 200},
    TestCase{25, 225, 200},
    TestCase{25, 675, 200},

    TestCase{50, 50, 100},
    TestCase{50, 150, 100},
    TestCase{50, 450, 100},
    TestCase{50, 1350, 100},

    TestCase{100, 100, 50},
    TestCase{100, 300, 50},
    TestCase{100, 900, 50},
    TestCase{100, 2700, 50},

    TestCase{200, 200, 25},
    TestCase{200, 600, 25},
    TestCase{200, 1800, 25},
    TestCase{200, 5400, 25},
};


using DMatrix = murty::DenseMatrix<double>;
using SMatrix = murty::SparseMatrix<double, int>;

void randomize_data(double* data, size_t n, double mn, double mx, std::mt19937& rng) {
    auto dist = std::uniform_real_distribution<double>(mn, mx);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(rng);
    }
}

DMatrix get_random_matrix(size_t rows, size_t cols, double mn, double mx, std::mt19937& rng) {
    DMatrix ret(rows, cols);
    randomize_data(ret.data(), rows * cols, mn, mx, rng);
    return ret;
}

DMatrix get_geom_matrix(size_t rows, size_t cols, double mn, double mx, std::mt19937& rng) {
    std::vector<double> R(2 * rows), C(2 * cols);
    randomize_data(R.data(), 2 * rows, mn, mx, rng);
    randomize_data(C.data(), 2 * cols, mn, mx, rng);
    DMatrix ret(rows, cols);
    double range = mx - mn;
    double max_d = std::sqrt(2) * range;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double dx = R[2 * i] - C[2 * j];
            double dy = R[2 * i + 1] - C[2 * j + 1];
            double dist = std::sqrt(dx * dx + dy * dy);
            ret(i, j) = mn + (dist / max_d) * range;
        }
    }
    return ret;
}

void print_matrix(const DMatrix& mat) {
    for (size_t i = 0; i < mat.rows(); i++) {
        for (size_t j = 0; j < mat.cols(); j++) {
            std::cout << mat(i, j) << ' ';
        }
        std::cout << std::endl;
    }
}

DMatrix get_reduced_matrix(const DMatrix& C, murty::SSPWorkers<double, int>& W) {
    size_t rows = C.rows(), cols = C.cols();
    murty::Subproblem<double, int> sol(0, rows, cols);
    sol.rows2use.resize(rows);
    std::iota(sol.rows2use.begin(), sol.rows2use.end(), 0);
    sol.cols2use.resize(cols);
    std::iota(sol.cols2use.begin(), sol.cols2use.end(), 0);
    full_ssp(sol, C, W);
    
    DMatrix ret(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        int matched_col = sol.col4row[i];
        double ui = (matched_col == sol.EMPTY ? 0 : C(i, matched_col) - sol.v[matched_col]);
        for (size_t j = 0; j < cols; j++) {
            ret(i, j) = C(i, j) - ui - sol.v[j];
        }
    }
    return ret;
}


SMatrix build_sparse(const DMatrix& C, const DMatrix& RC, size_t max_per_row) {

    max_per_row = std::min(max_per_row, C.cols());
    std::vector<int> row_idxs, col_idxs;
    std::vector<double> vals;
    row_idxs.push_back(0);

    std::vector<int> order(C.cols());
    std::iota(order.begin(), order.end(), 0);

    for (size_t row = 0; row < C.rows(); row++) {
        std::nth_element(order.begin(), order.begin() + max_per_row - 1, order.end(), [&](int x, int y){
            return RC(row, x) < RC(row, y);
        });
        std::sort(order.begin(), order.begin() + max_per_row);
        for (size_t i = 0; i < max_per_row; i++) {
            auto idx = order[i];
            col_idxs.push_back(idx);
            vals.push_back(C(row, idx));
        }
        row_idxs.push_back(vals.size());
    }
    return {C.cols(), vals, col_idxs, row_idxs};
}

size_t find_thr(
        const DMatrix& C, const DMatrix& order_C, size_t K,
        const std::vector<murty::Assignment<double, int>>& ass, murty::MurtyWorkers<double, int>& W) {
    double max_cost = ass.back().cost + 1e-6;
    size_t l = 0, r = C.rows();
    while (l < r) {
        size_t m = (l + r) / 2;
        SMatrix SC = build_sparse(C, order_C, m);
        auto ass_red = murty::solve_subsets<double, int>(SC, K, W, {}, {}, {}, max_cost);
        if (ass_red.size() == ass.size()) r = m;
        else l = m + 1;
    }
    return r;
}

void run(int t, double mn, double mx) {
    for (auto tc : tcs) {
        std::cout << "--------------------------------------------------\n";
        std::cout << "N = " << tc.N << ", K = " << tc.K << '\n';
        std::cout << "--------------------------------------------------\n";

        size_t mx_e = 0, sm_e = 0;
        size_t mx_g = 0, sm_g = 0;
        std::mt19937 rng(42);
        murty::SSPWorkers<double, int> W(tc.N, tc.N);
        murty::MurtyWorkers<double, int> WM(tc.K, tc.N, tc.N);

        for (size_t i = 0; i < tc.num; i++) {
            DMatrix C = (t == 1 ? get_random_matrix(tc.N, tc.N, mn, mx, rng) : get_geom_matrix(tc.N, tc.N, -1, 1, rng));
            DMatrix RC = get_reduced_matrix(C, W);
            auto ass_ref = murty::solve_subsets<double, int>(C, tc.K, WM);

            size_t thr_e = find_thr(C, RC, tc.K, ass_ref, WM);
            size_t thr_g = find_thr(C, C, tc.K, ass_ref, WM);
            mx_e = std::max(mx_e, thr_e);
            mx_g = std::max(mx_g, thr_g);
            sm_e += thr_e;
            sm_g += thr_g;
        }
        std::cout << "   (EXPERIMENT) MAX = " << mx_e << ", AVG = " << sm_e / tc.num << std::endl;
        std::cout << "   (GREEDY)     MAX = " << mx_g << ", AVG = " << sm_g / tc.num << std::endl;
    }
}

int main() {

    std::cout << "\n\n==================================================\n";
    std::cout << "   RANDOM, RANGE (-1, 1)\n";
    std::cout << "==================================================\n";
    run(1, -1, 1);

    std::cout << "\n\n==================================================\n";
    std::cout << "   RANDOM, RANGE (-1, 0)\n";
    std::cout << "==================================================\n";
    run(1, -1, 0);

    std::cout << "\n\n==================================================\n";
    std::cout << "   GEOM, RANGE (-1, 1)\n";
    std::cout << "==================================================\n";
    run(2, -1, 1);

    std::cout << "\n\n==================================================\n";
    std::cout << "   GEOM, RANGE (-1, 0)\n";
    std::cout << "==================================================\n";
    run(2, -1, 0);
}