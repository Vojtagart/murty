/**
 * @file      murty_split.hpp
 * @brief     Implements split of the subproblem for the murty's algortihm
 * @author    @vojtagart
 * @date      21/02/2026
 * @see       https://github.com/Vojtagart/murty
 * * @par Credits & Acknowledgments
 * - Algorithm inspired by: M. Motro and J. Ghosh, "Scaling Data Association for Hypothesis-Oriented MHT," 2019 22th International Conference on Information Fusion (FUSION), Ottawa, ON, Canada, 2019, pp. 1-8.
 * - Reference implementation: https://github.com/motrom/fastmurty/tree/master
 * - Documentation formatted and refined with the assistance of Google Gemini.
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <cassert>
#include <numeric>
#include <memory>
#include <ranges>
#include <functional>
#include <iostream>
#include <cstdint>
#include <cstddef>

#include "subproblem.hpp"
#include "matrix.hpp"
#include "sparse_matrix.hpp"
#include "interval_heap.hpp"
#include "ssp.hpp"


namespace murty {

/**
 * @brief Reusable worker buffers for Murty split operation
 *
 * Pre-allocates and reuses working storage to avoid repeated allocations
 * during the split process. Initialize with the maximum dimensions that
 * will be encountered
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 */
template <typename Scalar, typename Idx>
struct SplitWorkers {
    std::vector<Scalar> min_slack;              ///< Minimum slack per row
    std::vector<Scalar> min_slack_miss;         ///< Min slack considering augmenting/missing columns
    std::vector<Idx> best_col;                  ///< Best alternative column per row
    std::vector<Idx> cols2use;                  ///< Active columns in current iteration
    std::vector<uint8_t> col_in_use;            ///< Whether column is closed or not (for sparse)
    std::vector<Scalar> u;                      ///< Row reductions (dual variables)
    std::vector<Idx> col_mapping;               ///< Mapping of column to index in sol.cols2use

    /**
     * @brief Constructs the worker with buffers for given problem size
     *
     * @param rows Maximum number of rows
     * @param cols Maximum number of columns
     */
    constexpr SplitWorkers(size_t rows = 0, size_t cols = 0)
            : min_slack(rows), min_slack_miss(rows), best_col(rows), col_in_use(cols, false), u(rows), col_mapping(cols) {
        cols2use.reserve(cols);
    }
    /**
     * @brief Resizes the worker buffers to accommodate larger dimensions
     *
     * @param rows Target maximum number of rows
     * @param cols Target maximum number of columns
     */
    constexpr void resize(size_t rows, size_t cols) {
        if (min_slack.size() < rows) {
            min_slack.resize(rows);
            min_slack_miss.resize(rows);
            best_col.resize(rows);
            u.resize(rows);
        }
        if (col_in_use.size() < cols) {
            col_in_use.resize(cols, false);
            col_mapping.resize(cols);
        }
        cols2use.reserve(cols);
    }
};

//==========================================================================================

/**
 * @brief Executes the split operation of Murty's algorithm
 *
 * Generates new subproblems by systematically banning edges from the current optimal matching.
 * If a valid new subproblem has a cost better than the current worst in the heap, it solves
 * the subproblem (using SSP) and updates the heap of best solutions.
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam Matrix Type of the cost matrix
 * @tparam HeapT Type of the solution heap
 *
 * @param C The cost matrix
 * @param sol Working subproblem buffer to be manipulated and potentially store
 * @param ref_sol The reference subproblem containing the current optimal matching to split
 * @param sols Heap tracking the top k best solutions
 * @param W Pre-allocated worker buffers for the split operation
 * @param WSSP Pre-allocated worker buffers for the SSP algorithm
 */
template <typename Scalar, typename Idx, typename MatrixT, typename HeapT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
void murty_split(
        const MatrixT& C, Subproblem<Scalar, Idx>& worker, Subproblem<Scalar, Idx>& sol,
        HeapT& sols, SplitWorkers<Scalar, Idx>& W, SSPWorkers<Scalar, Idx>& WSSP);

/**
 * @brief Determines the optimal row splitting order based on slack estimates
 *
 * Orders the rows of the subproblem such that rows with the largest minimum slack are split first.
 * This heuristic maximizes the lower bound of generated subproblems to prune the search space early.
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam Matrix Type of the cost matrix
 *
 * @param C The cost matrix
 * @param sol The subproblem whose rows will be reordered
 * @param W Worker buffers for slack computation
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
void split_order(const MatrixT& C, Subproblem<Scalar, Idx>& sol, SplitWorkers<Scalar, Idx>& W);

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT, typename HeapT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
void murty_split(
        const MatrixT& C, Subproblem<Scalar, Idx>& worker, Subproblem<Scalar, Idx>& sol,
        HeapT& sols, SplitWorkers<Scalar, Idx>& W, SSPWorkers<Scalar, Idx>& WSSP) {

    using SolT = Subproblem<Scalar, Idx>;
    const Scalar INF = std::numeric_limits<Scalar>::max();

    if (sol.rows() == 0 || sol.cols() == 0) return;
    
    worker.copy_matching(sol);
    worker.cur_cost = sol.cur_cost;
    split_order(C, sol, W);
    Scalar max_cost = sols.max()->cur_cost;
    bool last = true;

    for (size_t j = 0; j < sol.cols(); j++) {
        W.col_mapping[sol.cols2use[j]] = static_cast<Idx>(j);
    }

    // In the for cycle: Assuming that sol has no bans inside
    // or it does, but then it's the last row
    while (sol.rows() > 0) {
        Idx row = sol.rows2use.back();
        Scalar cost = W.min_slack[row] + worker.cur_cost;
        Idx col = sol.col4row[row];
        bool rep = false;

        // if the new sobproblem created by splitting based on this row
        // has a potential to be among the k best, solve it
        if (cost < max_cost) {
            // ban matching for curent row - curently in the last spot
            if (col == SolT::EMPTY) {
                sol.allow_miss = false;
                // get back that extra added banned col from constructor
            } else {
                sol.banned_cols[col] = true;
            }
            if (ssp_last(sol, C, WSSP, max_cost - sol.cur_cost) < INF) {
                assert(sol.cur_cost < max_cost + EPS);
                SolT* tmp = sols.max();
                *tmp = sol;
                sols.replace_max(tmp);
                max_cost = sols.max()->cur_cost;
                rep = true;
            }
        }
        // fix the matching by removing the row and its column
        assert(col != SolT::UNMATCHED);
        sol.rows2use.pop_back();
        if (col != SolT::EMPTY) {
            size_t idx = W.col_mapping[col];
            std::swap(sol.cols2use[idx], sol.cols2use.back());
            W.col_mapping[sol.cols2use[idx]] = static_cast<Idx>(idx);
            sol.cols2use.pop_back();
        }

        if (last) {
            last = false;
            std::fill(sol.banned_cols.begin(), sol.banned_cols.end(), false);
        } else if (col != SolT::EMPTY) {
            sol.banned_cols[col] = false;
        }
        sol.allow_miss = true;

        // Insert back the original matching and cost
        if (rep && !sol.rows2use.empty()) {
            sol.copy_matching(worker);
            sol.cur_cost = worker.cur_cost;
        }
    }
}

//==========================================================================================

namespace internal {
/**
 * @brief Updates the minimum slack estimate for a given row
 *
 * Calculates the minimum cost difference (slack) to alternative columns, factoring in dual variables.
 * Optionally respects banned columns for the active splitting row.
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indices
 * @tparam Matrix Type of the cost matrix
 * @tparam CHECK_BANS Boolean flag to enable checking for banned columns
 *
 * @param row The index of the row to update
 * @param C The cost matrix
 * @param sol The current subproblem state
 * @param W Worker buffers storing the slacks and active columns
 */
template <typename Scalar, typename Idx, typename MatrixT, bool CHECK_BANS=false>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
void update_slack(Idx row, const MatrixT& C, const Subproblem<Scalar, Idx>& sol, SplitWorkers<Scalar, Idx>& W) {
    using SolT = Subproblem<Scalar, Idx>;
    constexpr bool IS_DENSE = is_matrix_v<MatrixT>;
    
    // Reseting slack, instead of INF we can use min_slack_miss
    W.min_slack[row] = W.min_slack_miss[row];
    W.best_col[row] = SolT::EMPTY;
    Idx matched_col = sol.col4row[row];
    if constexpr (IS_DENSE) {
        for (auto col : W.cols2use) {
            // ignore match column
            if (col == matched_col) continue;
            // for the last row, do not count banned cols
            // if should be optimized via the compiler
            if (CHECK_BANS && sol.banned_cols[col]) continue;
            Scalar slack = C(row, col) - sol.v[col] - W.u[row];
            assert(slack > -EPS && "Negative slack");
            if (slack < W.min_slack[row]) {
                W.min_slack[row] = slack;
                W.best_col[row] = col;
            }
        }
    } else {
        auto elems = C.row_elems(row);
        for (size_t j = 0; j < elems.size(); j++) {
            Idx col = elems[j].col;
            if (col == matched_col || !W.col_in_use[col]) continue;
            if (CHECK_BANS && sol.banned_cols[col]) continue;
            Scalar slack = elems[j].val - sol.v[col] - W.u[row];
            assert(slack > -EPS && "Negative slack");
            if (slack < W.min_slack[row]) {
                W.min_slack[row] = slack;
                W.best_col[row] = col;
            }
        }
    }
}

} // namesapce internal

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
void split_order(const MatrixT& C, Subproblem<Scalar, Idx>& sol, SplitWorkers<Scalar, Idx>& W) {
    using std::swap;
    using SolT = Subproblem<Scalar, Idx>;
    const Scalar INF = std::numeric_limits<Scalar>::max();
    constexpr bool IS_DENSE = is_matrix_v<MatrixT>;

    if (sol.rows() == 0) return;

    W.resize(sol.max_rows(), sol.max_cols());
    #ifndef NDEBUG
    for (size_t j = 0; j < sol.max_cols(); j++)
        assert(!W.col_in_use[j] && "W.col_in_use isnt initalized properly");
    #endif

    auto set_colinuse = [&](bool val){
        for (auto col : sol.cols2use) {
            W.col_in_use[col] = val;
        }
    };
    if constexpr (!IS_DENSE) {
        set_colinuse(true);
        Idx col = sol.col4row[sol.rows2use.back()];
        if (col != SolT::EMPTY)
            W.col_in_use[col] = false;
    }

    // ignoring the last row (placing it as the worst for easier banned_cols representation)
    size_t rows = sol.rows() - 1;
    W.cols2use.clear();

    // calculating the row reductions, including the last row
    for (auto row : sol.rows2use) {
        Idx matched_col = sol.col4row[row];
        assert(matched_col != SolT::UNMATCHED);
        if (matched_col == SolT::EMPTY) {
            W.u[row] = 0;
            W.min_slack_miss[row] = INF;
        } else {
            if constexpr (IS_DENSE) {
                W.u[row] = C(row, matched_col) - sol.v[matched_col];
            } else {
                auto elems = C.row_elems(row);
                auto it = elems.size() >= LB_THR ? std::ranges::lower_bound(elems, matched_col, std::less<Idx>{}, &MatrixT::Elem::col)
                                                 : std::ranges::find(elems, matched_col, &MatrixT::Elem::col);
                assert(it != elems.end() && it->col == matched_col && "Matched to unconnected column");
                W.u[row] = it->val - sol.v[matched_col];
            }
            // storing slack with augmening column
            W.min_slack_miss[row] = (row == sol.rows2use.back() && !sol.allow_miss ? INF : -W.u[row]);
            assert(W.min_slack_miss[row] > -EPS && "Slack must be possitive");
        }
    }

    if constexpr (IS_DENSE) {
        for (auto col : sol.cols2use) {
            // missed column will never be fixed, so the minimal slack
            // can always be to that column. Also, these dont need to be updated
            if (sol.row4col[col] == SolT::EMPTY) {
                for (auto row : sol.rows2use) {
                    // skip the banned cols (only for the last row)
                    if (row == sol.rows2use.back() && sol.banned_cols[col]) continue;
                    Scalar slack = C(row, col) - sol.v[col] - W.u[row];
                    assert(slack > -EPS && "Negative slack");
                    // since missed columns are never banned, slack for given row and missed column
                    // never changes, so we can store min of these values along with slack with aug. column
                    W.min_slack_miss[row] = std::min(W.min_slack_miss[row], slack);
                }
            // efectivelly banning column matched to the last row
            } else if (sol.row4col[col] != sol.rows2use.back()) {
                W.cols2use.push_back(col);
            }
        }
    }

    // calculate estimates for all rows
    for (size_t i = 0; i < rows; i++)
        internal::update_slack(sol.rows2use[i], C, sol, W);
    // Calculate slack estimate for the last row (with ban checking)
    internal::update_slack<Scalar, Idx, MatrixT, true>(sol.rows2use.back(), C, sol, W);

    // Iteratively select rows in order of increasing slack (worst to best)
    while (rows > 1) {
        // find row with the maximal minimum slacxk
        size_t idx = std::max_element(sol.rows2use.begin(), sol.rows2use.begin() + rows, [&](Idx x, Idx y){
            return W.min_slack[x] < W.min_slack[y];
        }) - sol.rows2use.begin();
        Idx row = sol.rows2use[idx];
        // set this row as k'th worst and "delete"
        --rows;
        std::swap(sol.rows2use[idx], sol.rows2use[rows]);

        Idx matched_col = sol.col4row[row];
        if (matched_col == SolT::EMPTY) continue;

        if constexpr (IS_DENSE) {
            auto it_match = std::ranges::find(W.cols2use, matched_col);
            assert(it_match != W.cols2use.end());
            // deleting the matched col for the rest
            *it_match = W.cols2use.back();
            W.cols2use.pop_back();
        } else {
            W.col_in_use[matched_col] = false;
        }
        // update slacks for all rows that used this slack
        for (size_t i = 0; i < rows; i++) {
            row = sol.rows2use[i];
            if (W.best_col[row] == matched_col)
                internal::update_slack(row, C, sol, W);
        }
    }
    set_colinuse(false);
}

} // namesapce murty