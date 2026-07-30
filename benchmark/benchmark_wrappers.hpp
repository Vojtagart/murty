#pragma once
#include <vector>
#include <chrono>
#include <numeric>
#include <span>
#include <algorithm>

#include "murty/solve.hpp"
#include "murty/sparse_matrix.hpp"

extern "C" {
    #include "da.h"
}

namespace murty::benchmark {

using SMatrix = SparseMatrix<double, int>;
using DMatrix = DenseMatrix<double>;
using DMatrixV = DenseMatrixView<double>;

struct DenseMatrixData {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<double> data;
};

struct BenchmarkResult {
    double time_us = 0.;
    std::vector<double> costs;
};

#ifdef SPARSE
// TODO
#else

DMatrixV data_to_dense(const DenseMatrixData& data) {
    return DMatrixV(data.data.data(), data.rows, data.cols);
}

BenchmarkResult run_murty_single(const DenseMatrixData& in) {
    murty::MurtyWorkers<double, int> W(1, in.rows, in.cols);
    auto C = data_to_dense(in);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = murty::assignment<double, int, DMatrixV>(C, W);
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    return {total, {result.cost}};
}

BenchmarkResult run_murty_k(const DenseMatrixData& in, size_t K) {
    murty::MurtyWorkers<double, int> W(K, in.rows, in.cols);
    auto C = data_to_dense(in);
    std::span<const DMatrixV> C_span(&C, 1);
    auto start = std::chrono::high_resolution_clock::now();
    auto [asss, idxs] = murty::solve<double, int, DMatrixV>(C_span, K, W);
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    std::vector<double> costs;
    for (const auto& ass : asss) {
        costs.push_back(ass.cost);
    }
    return {total, std::move(costs)};
}

BenchmarkResult run_fastmurty_k(const DenseMatrixData& in, size_t K) {
    WorkvarsforDA workvars = allocateWorkvarsforDA(static_cast<int>(in.rows), static_cast<int>(in.cols), static_cast<int>(K));
    std::vector<unsigned char> row_priors(in.rows, 1);
    std::vector<unsigned char> col_priors(in.cols, 1);
    std::vector<int> out_assocs(K * (in.rows + in.cols) * 2, 0);
    std::vector<double> out_costs(K, 0.), C(in.data);
    double weight = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    da(C.data(), 1, (bool*)row_priors.data(), &weight, 1, (bool*)col_priors.data(), &weight, 
       static_cast<int>(K), out_assocs.data(), out_costs.data(), &workvars);
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    deallocateWorkvarsforDA(workvars);
    while (!out_costs.empty() && out_costs.back() >= 1e9) {
        out_costs.pop_back();
    }
    return {total, std::move(out_costs)};
}

BenchmarkResult run_fastmurty_single(const DenseMatrixData& in) {
    return run_fastmurty_k(in, 1);
}

#endif




}
