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

SparseMatrixData get_sparse_data(size_t rows, size_t cols, double mn, double mx, double p, std::mt19937& rng) {
    auto dist = std::uniform_real_distribution<double>(mn, mx);
    auto flip = std::uniform_real_distribution<double>(0., 1.);
    std::vector<int> row_idxs = {0};
    std::vector<int> col_idxs;
    std::vector<double> vals;
    for (size_t i = 0; i < rows; i++) {
        bool ins = false;
        for (size_t j = 0; j < cols; j++) {
            double fl = flip(rng);
            // Keeping atleast 1 element per row since fastmurty segfaults with empty rows
            if (fl < p || (!ins && j == cols - 1)) {
                vals.push_back(dist(rng));
                col_idxs.push_back(j);
                ins = true;
            }
        }
        row_idxs.push_back(vals.size());
    }
    return {rows, cols, std::move(row_idxs), std::move(col_idxs), std::move(vals)};
}


int main() {
    CSVWriter csv("benchmark_sparse.csv");
    murty::MurtyWorkers<double, int> W;

    std::cout << "---------------------------------------------\n";
    std::cout << " single best assignment\n";
    std::cout << "---------------------------------------------\n";

    for (auto tc : tcs_single) {
        std::cout << "[ROWS = " << tc.rows << ", COLS = " << tc.cols << "]\n";

        std::vector<Procedure<SparseMatrixData>> prods = {
            {"Murty",     0., [&W](const SparseMatrixData& in, size_t K) -> BenchmarkResult {return run_murty_single(data_to_sparse(in), W);}},
            {"Fastmurty", 0.,   [](const SparseMatrixData& in, size_t K) -> BenchmarkResult {return run_fastmurty_single(in);}}
        };
        
        std::mt19937 rng(42);
        double p = 20. / tc.cols;
        for (size_t i = 0; i < tc.num; i++) {
            auto in = get_sparse_data(tc.rows, tc.cols, -1., 1., p, rng);
            run_procedures(prods, in);
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

        std::vector<Procedure<SparseMatrixData>> prods = {
            {"Murty",     0., [&W](const SparseMatrixData& in, size_t K) -> BenchmarkResult {return run_murty_k(data_to_sparse(in), K, W);}},
            {"Fastmurty", 0.,   [](const SparseMatrixData& in, size_t K) -> BenchmarkResult {return run_fastmurty_k(in, K);}}
        };

        std::mt19937 rng(42);
        double p = 20. / tc.cols;
        for (size_t i = 0; i < tc.num; i++) {
            auto in = get_sparse_data(tc.rows, tc.cols, -1., 1., p, rng);
            run_procedures(prods, in, tc.K);
        }
        report_procedures(prods, tc.num, [&csv, &tc](const std::string& name, double avg_time){
            csv.write_row("k_best", tc.rows, tc.cols, tc.K, tc.num, name, avg_time);
        });
        std::cout << std::endl;
    }
}