#include <iostream>
#include <iomanip>
#include <random>
#include "benchmark_wrappers.hpp"

using namespace murty::benchmark;


struct TestCase {
    size_t rows = 0, cols = 0, K = 0, num = 0;
};

std::vector<TestCase> tcs_single = {
    TestCase{10, 10, 1, 100},
    TestCase{25, 25, 1, 100},
    TestCase{50, 50, 1, 100},
    TestCase{100, 100, 1, 50},
    TestCase{200, 200, 1, 25},
    TestCase{300, 300, 1, 20},
    TestCase{500, 500, 1, 10}
};

DenseMatrixData get_dense_data(size_t rows, size_t cols, double mn, double mx) {
    std::mt19937 rng(42);
    auto dist = std::uniform_real_distribution<double>(mn, mx);
    std::vector<double> data(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] = dist(rng);
    }
    return {rows, cols, std::move(data)};
}

int main() {

    for (auto tc : tcs_single) {
        std::cout << "[ROWS = " << tc.rows << ", COLS = " << tc.cols << "]\n";
        double t_murty = 0., t_fastmurty = 0.;
        for (size_t i = 0; i < tc.num; i++) {
            auto in = get_dense_data(tc.rows, tc.cols, -1., 1.);
            auto r_murty = run_murty_single(in);
            auto r_fastmurty = run_murty_single(in);
            t_murty += r_murty.time_us;
            t_fastmurty += r_fastmurty.time_us;
            if (r_murty.costs.size() != r_fastmurty.costs.size())
                throw std::runtime_error("Costs sizes do not match");
            for (size_t j = 0; j < r_murty.costs.size(); j++) {
                if (std::abs(r_murty.costs[j] - r_fastmurty.costs[j]) > 1e-6)
                    throw std::runtime_error("Costs do not match");
            }
        }
        t_murty /= tc.num;
        t_fastmurty /= tc.num;
        std::cout << "Murty      : " << t_murty << " us\n";
        std::cout << "Fastmurty  : " << t_fastmurty << " us\n\n";
    }
}