/**
 * @file      benchmark_helpers.hpp
 * @brief     Implements helpers for running the benchmarks such as wrappers
 * @author    @vojtagart
 * @date      18/08/2026
 * @see       https://github.com/Vojtagart/murty
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini.
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <chrono>
#include <numeric>
#include <span>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <string>
#include "murty/solve.hpp"
#include "murty/sparse_matrix.hpp"

#ifdef FASTMURTY
extern "C" {
    #include "da.h"
}
#endif

namespace murty::benchmark {

using SMatrix = SparseMatrix<double, int>;
using DMatrix = DenseMatrix<double>;
using DMatrixV = DenseMatrixView<double>;

struct DenseMatrixData {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<double> data;
};

struct SparseMatrixData {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<int> row_idxs;
    std::vector<int> col_idxs;
    std::vector<double> vals;
};

struct BenchmarkResult {
    double time_us = 0.;
    std::vector<double> costs;
};

struct CSVWriter {
    std::ofstream file;
    CSVWriter(const std::string& filename) {
        file.open(filename);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file " + filename);
        file << "Tag,Rows,Cols,K,Num,Procedure,Time_us\n";
    }
    void write_row(const std::string& tag, size_t rows, size_t cols, size_t K, size_t num, const std::string& proc, double time_us) {
        file << tag << ',' << rows << ',' << cols << ',' << K << ',' << num << ',' << proc << ',' << time_us << '\n';
    }
};

SMatrix data_to_sparse(const SparseMatrixData& data) {
    return SMatrix(data.cols, data.vals, data.col_idxs, data.row_idxs);
}

#if defined(FASTMURTY) && defined(SPARSE)
cs_di_sparse data_to_sparse_fm(const SparseMatrixData& data) {
    cs_di_sparse c_sparse;
    c_sparse.m = static_cast<int>(data.rows);
    c_sparse.n = static_cast<int>(data.cols);
    c_sparse.nz = static_cast<int>(data.vals.size());
    c_sparse.nzmax = static_cast<int>(data.vals.size());
    c_sparse.p = const_cast<int*>(data.row_idxs.data());
    c_sparse.i = const_cast<int*>(data.col_idxs.data());
    c_sparse.x = const_cast<double*>(data.vals.data());
    return c_sparse;
}
#endif

BenchmarkResult run_murty_single(const SparseMatrixData& in, murty::MurtyWorkers<double, int>& W) {
    W.resize(1, in.rows, in.cols);
    auto C = data_to_sparse(in);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = murty::assignment<double, int, SMatrix>(C, W);
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    return {total, {result.cost}};
}

BenchmarkResult run_murty_k(const SparseMatrixData& in, size_t K, murty::MurtyWorkers<double, int>& W) {
    W.resize(K, in.rows, in.cols);
    auto C = data_to_sparse(in);
    std::span<const SMatrix> C_span(&C, 1);
    auto start = std::chrono::high_resolution_clock::now();
    auto [asss, idxs] = murty::solve<double, int, SMatrix>(C_span, K, W);
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    std::vector<double> costs;
    for (const auto& ass : asss) {
        costs.push_back(ass.cost);
    }
    return {total, std::move(costs)};
}

#if defined(FASTMURTY) && defined(SPARSE)
BenchmarkResult run_fastmurty_k(const SparseMatrixData& in, size_t K) {
    WorkvarsforDA workvars = allocateWorkvarsforDA(static_cast<int>(in.rows), static_cast<int>(in.cols), static_cast<int>(K));
    std::vector<unsigned char> row_priors(in.rows, 1);
    std::vector<unsigned char> col_priors(in.cols, 1);
    std::vector<int> out_assocs(K * (in.rows + in.cols) * 2, 0);
    std::vector<double> out_costs(K, 0.);
    double weight = 0.0;
    cs_di_sparse C = data_to_sparse_fm(in);

    auto start = std::chrono::high_resolution_clock::now();
    da(C, 1, (bool*)row_priors.data(), &weight, 1, (bool*)col_priors.data(), &weight, 
       static_cast<int>(K), out_assocs.data(), out_costs.data(), &workvars);
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    deallocateWorkvarsforDA(workvars);
    while (!out_costs.empty() && out_costs.back() >= 1e9) {
        out_costs.pop_back();
    }
    return {total, std::move(out_costs)};
}

BenchmarkResult run_fastmurty_single(const SparseMatrixData& in) {
    return run_fastmurty_k(in, 1);
}
#endif


DMatrixV data_to_dense(const DenseMatrixData& data) {
    return DMatrixV(data.data.data(), data.rows, data.cols);
}

BenchmarkResult run_murty_single(const DenseMatrixData& in, murty::MurtyWorkers<double, int>& W) {
    W.resize(1, in.rows, in.cols);
    auto C = data_to_dense(in);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = murty::assignment<double, int, DMatrixV>(C, W);
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    return {total, {result.cost}};
}

BenchmarkResult run_murty_k(const DenseMatrixData& in, size_t K, murty::MurtyWorkers<double, int>& W) {
    W.resize(K, in.rows, in.cols);
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

#if defined(FASTMURTY) && !defined(SPARSE)
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

template <typename MatrixT>
requires (murty::is_matrix_v<MatrixT> || murty::is_sparse_matrix_v<MatrixT>)
BenchmarkResult run_murty_single(const MatrixT& C, murty::MurtyWorkers<double, int>& W) {
    W.resize(1, C.rows(), C.cols());
    auto start = std::chrono::high_resolution_clock::now();
    auto result = murty::assignment<double, int, MatrixT>(C, W);
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    return {total, {result.cost}};
}

template <typename MatrixT>
requires (murty::is_matrix_v<MatrixT> || murty::is_sparse_matrix_v<MatrixT>)
BenchmarkResult run_murty_k(const MatrixT& C, size_t K, murty::MurtyWorkers<double, int>& W) {
    W.resize(K, C.rows(), C.cols());
    std::span<const MatrixT> C_span(&C, 1);
    auto start = std::chrono::high_resolution_clock::now();
    auto [asss, idxs] = murty::solve<double, int, MatrixT>(C_span, K, W);
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    std::vector<double> costs;
    for (const auto& ass : asss) {
        costs.push_back(ass.cost);
    }
    return {total, std::move(costs)};
}

void compare_costs(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        std::cout << "Sizes are " << a.size() << " and " << b.size() << std::endl;
        throw std::runtime_error("Costs sizes do not match");
    }
    for (size_t j = 0; j < a.size(); j++) {
        if (std::abs(a[j] - b[j]) > 1e-6) {
            std::cout << "Mismatch on the " << j << "-th cost, " << a[j] << " vs " << b[j] << std::endl;
            throw std::runtime_error("Costs do not match");
        }
    }
}

} // namespace murty::benchmark
