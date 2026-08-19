#include <iostream>
#include <iomanip>
#include <random>
#define FASTMURTY
#include "benchmark_helpers.hpp"

using namespace murty::benchmark;


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

DenseMatrixData get_dense_data(size_t rows, size_t cols, double mn, double mx, std::mt19937& rng) {
    auto dist = std::uniform_real_distribution<double>(mn, mx);
    std::vector<double> data(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] = dist(rng);
    }
    return {rows, cols, std::move(data)};
}

int main() {
    CSVWriter csv("benchmark_dense.csv");
    murty::MurtyWorkers<double, int> W;

    std::cout << "---------------------------------------------\n";
    std::cout << " single best assignment\n";
    std::cout << "---------------------------------------------\n";

    for (auto tc : tcs_single) {
        std::cout << "[ROWS = " << tc.rows << ", COLS = " << tc.cols << "]\n";
        double t_murty = 0., t_fastmurty = 0.;
        std::mt19937 rng(42);
        for (size_t i = 0; i < tc.num; i++) {
            auto in = get_dense_data(tc.rows, tc.cols, -1., 1., rng);
            auto r_murty = run_murty_single(in, W);
            auto r_fastmurty = run_fastmurty_single(in);
            t_murty += r_murty.time_us;
            t_fastmurty += r_fastmurty.time_us;
            compare_costs(r_murty.costs, r_fastmurty.costs);
        }
        t_murty /= tc.num;
        t_fastmurty /= tc.num;
        std::cout << "Murty      : " << t_murty << " us\n";
        std::cout << "Fastmurty  : " << t_fastmurty << " us\n\n";
        csv.write_row("single", tc.rows, tc.cols, tc.K, tc.num, "Murty", t_murty);
        csv.write_row("single", tc.rows, tc.cols, tc.K, tc.num, "Fastmurty", t_fastmurty);
    }

    std::cout << "---------------------------------------------\n";
    std::cout << " K-best assigments\n";
    std::cout << "---------------------------------------------\n";

    for (auto tc : tcs_k) {
        std::cout << "[ROWS = " << tc.rows << ", COLS = " << tc.cols << ", K = " << tc.K << "]\n";
        double t_murty = 0., t_fastmurty = 0.;
        std::mt19937 rng(42);
        for (size_t i = 0; i < tc.num; i++) {
            auto in = get_dense_data(tc.rows, tc.cols, -1., 1., rng);
            auto r_murty = run_murty_k(in, tc.K, W);
            auto r_fastmurty = run_fastmurty_k(in, tc.K);
            t_murty += r_murty.time_us;
            t_fastmurty += r_fastmurty.time_us;
            compare_costs(r_murty.costs, r_fastmurty.costs);
        }
        t_murty /= tc.num;
        t_fastmurty /= tc.num;
        std::cout << "Murty      : " << t_murty << " us\n";
        std::cout << "Fastmurty  : " << t_fastmurty << " us\n\n";
        csv.write_row("k_best", tc.rows, tc.cols, tc.K, tc.num, "Murty", t_murty);
        csv.write_row("k_best", tc.rows, tc.cols, tc.K, tc.num, "Fastmurty", t_fastmurty);
    }
}