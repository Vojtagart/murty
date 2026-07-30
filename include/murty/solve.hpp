/**
 * @file      solve.hpp
 * @brief     Implements k-best assignment solvers usign murty's algorithm with SSP
 * @author    @vojtagart
 * @date      21/02/2026
 * @see       https://github.com/Vojtagart/murty
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini.
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <optional>
#include <iostream>
#include <memory>
#include <cassert>
#include <numeric>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <variant>
#include <span>

#include "matrix.hpp"
#include "sparse_matrix.hpp"
#include "subproblem.hpp"
#include "interval_heap.hpp"
#include "ssp.hpp"
#include "murty_split.hpp"


namespace murty {

/**
 * @brief Represents a single assignment solution and its total cost
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 */
template <typename Scalar, typename Idx>
struct Assignment {

    constexpr static Idx EMPTY = Subproblem<Scalar, Idx>::EMPTY;
    constexpr static Idx UNMATCHED = Subproblem<Scalar, Idx>::UNMATCHED;

    std::vector<std::pair<Idx, Idx>> ass;            ///< The pairs [row, col] in assignment
    Scalar cost;                                     ///< The total cost of the assignment

    Assignment() = default;
    Assignment(std::vector<std::pair<Idx, Idx>> ass, Scalar cost)
            : ass(std::move(ass)), cost(cost) {}
    explicit Assignment(const Subproblem<Scalar, Idx>& sol, size_t rows, size_t cols)
            : cost(sol.cur_cost) {
        assert(rows <= sol.max_rows());
        assert(cols <= sol.max_cols());

        ass.reserve(rows + cols);
        for (size_t i = 0; i < rows; i++) {
            Idx col = sol.col4row[i];
            assert(col != UNMATCHED);
            ass.emplace_back(static_cast<Idx>(i), col);
            if (col != EMPTY)
                assert(sol.row4col[col] == static_cast<Idx>(i));
        }
        for (size_t j = 0; j < cols; j++) {
            Idx row = sol.row4col[j];
            assert(row != UNMATCHED);
            if (row == EMPTY)
                ass.emplace_back(row, static_cast<Idx>(j));
        }
    }
};

//==========================================================================================

/**
 * @brief Bundled reusable worker buffers for Murty's algorithm
 *
 * Encapsulates worker buffers for SSP and splitting operations,
 * along with the priority queue and solution storage
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 */
template <typename Scalar, typename Idx>
struct MurtyWorkers {
    using SolT = Subproblem<Scalar, Idx>;

    struct CmpSol {
        constexpr bool operator()(const SolT* x, const SolT* y) const noexcept {
            return x->cur_cost < y->cur_cost;
        }
    };
    SSPWorkers<Scalar, Idx> ssp_w;
    SplitWorkers<Scalar, Idx> split_w;
    IntervalHeap<SolT*, std::vector<SolT*>, CmpSol> q;
    std::vector<SolT> sols;

    /**
     * @brief Initializes worker structures with reserved capacities
     *
     * @param K The number of best assignments to find
     * @param rows Initial maximum number of rows
     * @param cols Initial maximum number of columns
     */
    MurtyWorkers(size_t K = 0, size_t rows = 0, size_t cols = 0) : ssp_w(rows, cols), split_w(rows, cols) {
        q.reserve(K);
        sols.reserve(K + 1);
        for (size_t i = 0; i < K + 1; i++) {
            sols.emplace_back(0, rows, cols);
        }
    } 
    /**
     * @brief Resizes worker structures if larger capacities are needed
     *
     * @param K The target number of best assignments
     * @param rows Target maximum number of rows
     * @param cols Target maximum number of columns
     */
    void resize(size_t K, size_t rows, size_t cols) {
        ssp_w.resize(rows, cols);
        split_w.resize(rows, cols);
        q.reserve(K);
        sols.reserve(K + 1);

        for (auto& x : sols) {
            x.resize(rows, cols);
        }
        while (sols.size() < K + 1) {
            sols.emplace_back(0, rows, cols);
        }
    }
};

//==========================================================================================

template <typename Scalar, typename Idx=int>
using MatVariant = std::variant<
    DenseMatrix<Scalar>,
    DenseMatrixView<Scalar>,
    MatrixView<Scalar, Idx>,
    TransposedView<DenseMatrix<Scalar>, Scalar>,
    TransposedView<DenseMatrixView<Scalar>, Scalar>,
    TransposedView<MatrixView<Scalar, Idx>, Scalar>,
    SparseMatrix<Scalar, Idx>,
    SparseMatrixView<Scalar, Idx>
>;

//==========================================================================================

/**
 * @brief Solves for the K best assignments across multiple cost matrices
 *
 * Implements Murty's algorithm to find the K assignments with minimal cost
 * from a given set of cost matrices. Reuses the provided worker buffers.
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam MatrixT Type of the cost matrix
 *
 * @param C Vector of cost matrices
 * @param K Number of best assignments to find
 * @param W Pre-allocated worker buffers
 * @param base_costs Base cost for each input matrix
 * @param max_cost Maximal allowed cost of the matching
 *
 * @return A pair containing the K assignments and a vector indicating which matrix each assignment corresponds to
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT> || std::is_same_v<MatrixT, MatVariant<Scalar, Idx>>)
std::pair<std::vector<Assignment<Scalar, Idx>>, std::vector<Idx>> solve(
        std::span<const MatrixT> C, size_t K, MurtyWorkers<Scalar, Idx>& W, std::span<const Scalar> base_costs = {},
        Scalar max_cost = std::numeric_limits<Scalar>::max());

/**
 * @brief Solves for the K best assignments across multiple dense cost matrices
 *
 * Allocates local worker buffers and delegates to the main solver
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam MatrixT Type of the cost matrix
 *
 * @param C Vector of dense cost matrices
 * @param K Number of best assignments to find
 * @param base_costs Base cost for each input matrix
 * @param max_cost Maximal allowed cost of the matching
 *
 * @return A pair containing the K assignments and their corresponding matrix indices
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT> || std::is_same_v<MatrixT, MatVariant<Scalar, Idx>>)
std::pair<std::vector<Assignment<Scalar, Idx>>, std::vector<Idx>> solve(
        std::span<const MatrixT> C, size_t K, std::span<const Scalar> base_costs = {}, Scalar max_cost = std::numeric_limits<Scalar>::max()) {
    MurtyWorkers<Scalar, Idx> W(K);
    return solve(C, K, W, base_costs, max_cost);
}

//------------------------------------------------------------------------------------------

/**
 * @brief Solves for the K best assignments using a dense matrix with subsetting
 *
 * Reuses provided worker buffers. Can define multiple subproblems by providing
 * subsets of rows and/or columns to use.
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam MatrixT Type of the cost matrix
 *
 * @param C The base cost matrix
 * @param K Number of best assignments to find
 * @param W Pre-allocated worker buffers
 * @param row_subsets Optional subsets of rows defining distinct subproblems
 * @param col_subsets Optional subsets of columns defining distinct subproblems
 * @param base_costs Base cost for each subset
 * @param max_cost Maximal allowed cost of the matching
 *
 * @return vector of K best assignment
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT>)
std::vector<Assignment<Scalar, Idx>> solve_subsets(
        const MatrixT& C, size_t K, MurtyWorkers<Scalar, Idx>& W, const std::vector<std::vector<Idx>>& row_subsets = {},
        const std::vector<std::vector<Idx>>& col_subsets = {}, const std::vector<Scalar>& base_costs = {}, Scalar max_cost = std::numeric_limits<Scalar>::max());

/**
 * @brief Solves for the K best assignments using a dense matrix with subsetting
 *
 * Allocates local worker buffers.
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam MatrixT Type of the cost matrix
 *
 * @param C The base cost matrix
 * @param K Number of best assignments to find
 * @param row_subsets Optional subsets of rows defining distinct subproblems
 * @param col_subsets Optional subsets of columns defining distinct subproblems
 * @param base_costs Base cost for each subset
 * @param max_cost Maximal allowed cost of the matching
 *
 * @return vector of K best assignment
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT>)
std::vector<Assignment<Scalar, Idx>> solve_subsets(
        const MatrixT& C, size_t K, const std::vector<std::vector<Idx>>& row_subsets = {},
        const std::vector<std::vector<Idx>>& col_subsets = {}, const std::vector<Scalar>& base_costs = {}, Scalar max_cost = std::numeric_limits<Scalar>::max()) {
    MurtyWorkers<Scalar, Idx> W(K);
    return solve_subsets(C, K, W, row_subsets, col_subsets, base_costs, max_cost);
}

//------------------------------------------------------------------------------------------

/**
 * @brief Finds the single optimal assignment for a given cost matrix
 *
 * Uses SSP with provided worker buffers to compute the minimum cost perfect matching
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam MatrixT Type of the cost matrix
 *
 * @param C The cost matrix
 * @param W Pre-allocated worker buffers
 * @param max_cost Maximal allowed cost of the matching
 *
 * @return The optimal Assignment
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT>)
Assignment<Scalar, Idx> assignment(const MatrixT& C, SSPWorkers<Scalar, Idx>& W, Scalar max_cost = std::numeric_limits<Scalar>::max());

/**
 * @brief Finds the single optimal assignment utilizing Murty worker buffers
 *
 * Convenience overload that extracts the SSP workers from the broader Murty buffers
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam MatrixT Type of the cost matrix
 *
 * @param C The cost matrix
 * @param W Pre-allocated Murty worker buffers
 * @param max_cost Maximal allowed cost of the matching
 *
 * @return The optimal Assignment
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT>)
Assignment<Scalar, Idx> assignment(const MatrixT& C, MurtyWorkers<Scalar, Idx>& W, Scalar max_cost = std::numeric_limits<Scalar>::max()) {
    return assignment(C, W.ssp_w, max_cost);
}

/**
 * @brief Finds the single optimal assignment for a dense cost matrix
 *
 * Allocates local worker buffers
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam MatrixT Type of the cost matrix
 *
 * @param C The dense cost matrix
 * @param max_cost Maximal allowed cost of the matching
 *
 * @return The optimal Assignment
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT>)
auto assignment(const MatrixT& C, Scalar max_cost = std::numeric_limits<Scalar>::max()) {
    SSPWorkers<Scalar, Idx> W;
    return assignment(C, W, max_cost);
}

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT> || std::is_same_v<MatrixT, MatVariant<Scalar, Idx>>)
std::pair<std::vector<Assignment<Scalar, Idx>>, std::vector<Idx>> solve(
        std::span<const MatrixT> CS, size_t K, MurtyWorkers<Scalar, Idx>& W, std::span<const Scalar> base_costs, Scalar max_cost) {
    constexpr bool is_variant = std::is_same_v<MatrixT, MatVariant<Scalar, Idx>>;
    
    bool has_base = !base_costs.empty();
    if (has_base && base_costs.size() != CS.size())
        throw std::runtime_error("Base_costs size do not match the number of problems");

    Scalar INF = max_cost;
    std::pair<std::vector<Assignment<Scalar, Idx>>, std::vector<Idx>> ret;
    if (K == 0 || CS.empty()) return ret;

    auto& ass = ret.first; ass.reserve(K);
    auto& idxs = ret.second; idxs.reserve(K);

    size_t max_rows = 0;
    size_t max_cols = 0;
    for (const auto& C : CS) {
        if constexpr (is_variant) {
            max_rows = std::max(max_rows, std::visit([](const auto& m){return m.rows();}, C));
            max_cols = std::max(max_cols, std::visit([](const auto& m){return m.cols();}, C));
        } else {
            max_rows = std::max(max_rows, C.rows());
            max_cols = std::max(max_cols, C.cols());
        }
    }

    W.resize(K, max_rows, max_cols);
    W.q.clear();
    for (size_t i = 0; i < K; i++) {
        W.sols[i].cur_cost = INF;
        W.q.push(&(W.sols[i]));
    }

    // Worker solution, the one not in queue
    auto* wsol = &(W.sols.back()); 
    for (size_t i = 0; i < CS.size(); i++) {
        const auto& C = CS[i];

        size_t rows = 0, cols = 0;
        if constexpr (is_variant) {
            rows = std::visit([](const auto& m) {return m.rows();}, C);
            cols = std::visit([](const auto& m) {return m.cols();}, C);
        } else {
            rows = C.rows();
            cols = C.cols();
        }

        wsol->rows2use.resize(rows);
        std::iota(wsol->rows2use.begin(), wsol->rows2use.end(), 0);
        wsol->cols2use.resize(cols);
        std::iota(wsol->cols2use.begin(), wsol->cols2use.end(), 0);
        wsol->matrix_idx = i;
        
        Scalar base = (has_base ? base_costs[i] : 0);
        max_cost = W.q.max()->cur_cost;

        #ifndef NDEBUG
        wsol->base_cost = base;
        #endif

        Scalar cost = 0;
        if constexpr (is_variant)
            cost = std::visit([&](const auto& m) {return full_ssp(*wsol, m, W.ssp_w, max_cost, base);}, C);
        else
            cost = full_ssp(*wsol, C, W.ssp_w, max_cost, base);

        if (cost >= max_cost) continue;

        auto* tmp = W.q.max();
        W.q.replace_max(wsol);
        wsol = tmp;
    }

    for (size_t k = 0; k < K; k++) {
        auto* best_sol = W.q.min();
        W.q.pop_min();
        if (best_sol->cur_cost >= INF)
            break;

        size_t rows = 0, cols = 0;
        if constexpr (is_variant) {
            rows = std::visit([](const auto& m) {return m.rows();}, CS[best_sol->matrix_idx]);
            cols = std::visit([](const auto& m) {return m.cols();}, CS[best_sol->matrix_idx]);
        } else {
            rows = CS[best_sol->matrix_idx].rows();
            cols = CS[best_sol->matrix_idx].cols();
        }
        ass.emplace_back(*best_sol, rows, cols);
        idxs.push_back(best_sol->matrix_idx);

        if (W.q.empty()) break;
        if constexpr (is_variant)
            std::visit([&](const auto& m){murty_split(m, *wsol, *best_sol, W.q, W.split_w, W.ssp_w);}, CS[best_sol->matrix_idx]);
        else
            murty_split(CS[best_sol->matrix_idx], *wsol, *best_sol, W.q, W.split_w, W.ssp_w);
    }
    return ret;
}

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT>)
std::vector<Assignment<Scalar, Idx>> solve_subsets(
        const MatrixT& C, size_t K, MurtyWorkers<Scalar, Idx>& W, const std::vector<std::vector<Idx>>& row_subsets,
        const std::vector<std::vector<Idx>>& col_subsets, const std::vector<Scalar>& base_costs, Scalar max_cost) {

    Scalar INF = max_cost;
    using SolT = Subproblem<Scalar, Idx>;
    
    bool def_rows = row_subsets.empty();
    bool def_cols = col_subsets.empty();
    bool has_base = !base_costs.empty();
    size_t n = std::max({size_t(1), row_subsets.size(), col_subsets.size()});

    if (!def_cols && !def_rows && col_subsets.size() != 1 && row_subsets.size() != 1 && row_subsets.size() != col_subsets.size())
        throw std::runtime_error("Row and column subsets must have the same size or atleast one of them must have size 1");
    if (has_base && base_costs.size() != n)
        throw std::runtime_error("Base_costs size do not match the subsets sizes");
    
    // Special case - solve for only the given matrix
    if (col_subsets.empty() && row_subsets.empty()) {
        return solve<Scalar, Idx, MatrixT>(std::span<const MatrixT>(&C, 1), K, W, base_costs, max_cost).first;
    }

    // initialize default subsets - all the rows or all the cols
    std::vector<Idx> def_row_subs, def_col_subs;
    if (def_rows) {
        def_row_subs.resize(C.rows());
        std::iota(def_row_subs.begin(), def_row_subs.end(), 0);
    }
    if (def_cols) {
        def_col_subs.resize(C.cols());
        std::iota(def_col_subs.begin(), def_col_subs.end(), 0);
    }

    std::vector<Assignment<Scalar, Idx>> ret;
    if (K == 0) return ret;
    ret.reserve(K);

    W.resize(K, C.rows(), C.cols());
    W.q.clear();
    for (size_t i = 0; i < K; i++) {
        W.sols[i].cur_cost = INF;
        W.q.push(&(W.sols[i]));
    }

    // Worker solution, the one not in queue
    auto* wsol = &(W.sols.back()); 
    for (size_t i = 0; i < n; i++) {
        const auto& row = def_rows ? def_row_subs : (row_subsets.size() == 1 ? row_subsets[0] : row_subsets[i]);
        const auto& col = def_cols ? def_col_subs : (col_subsets.size() == 1 ? col_subsets[0] : col_subsets[i]);

        wsol->rows2use = row;
        wsol->cols2use = col;
        wsol->matrix_idx = i;
        
        Scalar base = (has_base ? base_costs[i] : 0);
        max_cost = W.q.max()->cur_cost;

        #ifndef NDEBUG
        wsol->base_cost = base;
        #endif

        Scalar cost = 0;
        cost = full_ssp(*wsol, C, W.ssp_w, max_cost, base);
        if (cost >= max_cost) continue;

        auto* tmp = W.q.max();
        W.q.replace_max(wsol);
        wsol = tmp;
    }

    for (size_t k = 0; k < K; k++) {
        auto* best_sol = W.q.min();
        W.q.pop_min();
        if (best_sol->cur_cost >= INF)
            break;

        size_t subs_idx = best_sol->matrix_idx;
        const auto& row = def_rows ? def_row_subs : (row_subsets.size() == 1 ? row_subsets[0] : row_subsets[subs_idx]);
        const auto& col = def_cols ? def_col_subs : (col_subsets.size() == 1 ? col_subsets[0] : col_subsets[subs_idx]);

        std::vector<std::pair<Idx, Idx>> ass;
        ass.reserve(row.size() + col.size());
        for (auto r : row) {
            Idx c = best_sol->col4row[r];
            assert(c != SolT::UNMATCHED);
            ass.emplace_back(r, c);
            if (c != SolT::EMPTY)
                assert(best_sol->row4col[c] == r);
        }
        for (auto c : col) {
            Idx r = best_sol->row4col[c];
            assert(r != SolT::UNMATCHED);
            if (r == SolT::EMPTY)
                ass.emplace_back(r, c);
        }
        ret.emplace_back(std::move(ass), best_sol->cur_cost);

        if (W.q.empty()) break;
        murty_split(C, *wsol, *best_sol, W.q, W.split_w, W.ssp_w);
    }
    return ret;
}

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT> || is_sparse_matrix_v<MatrixT>)
Assignment<Scalar, Idx> assignment(const MatrixT& C, SSPWorkers<Scalar, Idx>& W, Scalar max_cost) {

    size_t rows = C.rows(), cols = C.cols();
    Subproblem<Scalar, Idx> sol(0, rows, cols);
    sol.rows2use.resize(rows);
    std::iota(sol.rows2use.begin(), sol.rows2use.end(), 0);
    sol.cols2use.resize(cols);
    std::iota(sol.cols2use.begin(), sol.cols2use.end(), 0);

    if (full_ssp(sol, C, W, max_cost) >= max_cost)
        return Assignment<Scalar, Idx>(std::vector<std::pair<Idx, Idx>>{}, std::numeric_limits<Scalar>::max());
    return Assignment<Scalar, Idx>(sol, rows, cols);
}

} // namespace murty