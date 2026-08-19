#include <iostream>
#include <iomanip>
#include <random>
#define FASTMURTY
#define TCL_MURTY
#define FBAEUERLEIN_MURTY
#include "benchmark_helpers.hpp"

using namespace murty::benchmark;

constexpr bool VALIDATE = true;

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

        std::vector<Procedure<DenseMatrixData>> prods = {
            {"Murty",       0., [&W](const DenseMatrixData& in, size_t K) -> BenchmarkResult {return run_murty_single(data_to_dense(in), W);}},
            {"Fastmurty",   0.,   [](const DenseMatrixData& in, size_t K) -> BenchmarkResult {return run_fastmurty_single(in);}},
            {"Tcl",         0.,   [](const DenseMatrixData& in, size_t K) -> BenchmarkResult {return run_tcl_murty_k(in, 1);}}
        };

        std::mt19937 rng(42);
        for (size_t i = 0; i < tc.num; i++) {
            auto in = get_dense_data(tc.rows, tc.cols, -2., -1., rng);
            run_procedures(prods, in, 1, VALIDATE);
        }
        report_procedures(prods, tc.num, [&csv, &tc](const std::string& name, double avg_time){
            csv.write_row("single", tc.rows, tc.cols, tc.K, tc.num, name, avg_time);
        });
        std::cout << std::endl;
    }

    std::cout << "---------------------------------------------\n";
    std::cout << " K-best assigments\n";
    std::cout << "---------------------------------------------\n";

    for (auto tc : tcs_k) {
        std::cout << "[ROWS = " << tc.rows << ", COLS = " << tc.cols << ", K = " << tc.K << "]\n";

        std::vector<Procedure<DenseMatrixData>> prods = {
            {"Murty",       0., [&W](const DenseMatrixData& in, size_t K) -> BenchmarkResult {return run_murty_k(data_to_dense(in), K, W);}},
            {"Fastmurty",   0.,   [](const DenseMatrixData& in, size_t K) -> BenchmarkResult {return run_fastmurty_k(in, K);}},
            {"Tcl",         0.,   [](const DenseMatrixData& in, size_t K) -> BenchmarkResult {return run_tcl_murty_k(in, K);}}
        };

        std::mt19937 rng(42);
        for (size_t i = 0; i < tc.num; i++) {
            auto in = get_dense_data(tc.rows, tc.cols, -2., -1., rng);
            run_procedures(prods, in, tc.K, VALIDATE);
        }
        report_procedures(prods, tc.num, [&csv, &tc](const std::string& name, double avg_time){
            csv.write_row("k_best", tc.rows, tc.cols, tc.K, tc.num, name, avg_time);
        });
        std::cout << std::endl;
    }
}