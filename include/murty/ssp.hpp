/**
 * @file      ssp.hpp
 * @brief     Implements SSP to find (non-maximal cardinality) min-cost matching for given subproblems
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
#include <limits>
#include <numeric>
#include <span>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <cstddef>
#include <ranges>

#include "matrix.hpp"
#include "subproblem.hpp"
#include "sparse_matrix.hpp"
#include "binary_heap.hpp"


namespace murty {

constexpr size_t LB_THR = 64;   ///< threshold for using lower_bound to find the matched column in sparse case

namespace internal {
template <typename Scalar, typename Idx>
struct QElem {
    Scalar cost;
    Idx col;
    constexpr bool operator < (const QElem& other) const noexcept {
        return cost < other.cost;
    }
};
}

/**
 * @brief Reusable worker buffers for the SSP algorithm
 *
 * Pre-allocates and reuses working storage to avoid repeated allocations
 * during SSP computations. Initialize with the maximum dimensions that
 * will be used
 *
 * @tparam Scalar Type used in the cost matrix
 * @tparam Idx Type used for indexes
 */
template <typename Scalar, typename Idx>
struct SSPWorkers {
    std::vector<Idx> rows2use;                             ///< Active rows in current iteration
    std::vector<Idx> cols2use;                             ///< Active columns in current iteration (closed columns in sparse mode)
    std::vector<Scalar> dist;                              ///< Shortest distance to the columns
    std::vector<Idx> par;                                  ///< Previous row on the shortest path to column
    std::vector<uint8_t> col_in_use;                       ///< Whether is column closed or open (for sparse dijkstra)
    BinaryHeap<internal::QElem<Scalar, Idx>> heap;         ///< Binary heap for sparse mode

    /**
     * @brief Constructs worker buffers with given initial capacities
     *
     * @param rows Initial maximum number of rows
     * @param cols Initial maximum number of columns
     */
    constexpr SSPWorkers(size_t rows = 0, size_t cols = 0)
        : dist(cols), par(cols), col_in_use(cols, false) {
        rows2use.reserve(rows);
        cols2use.reserve(cols);
        heap.reserve(cols);
    }
    /**
     * @brief Resizes the worker buffers if the new dimensions exceed current capacity
     *
     * @param rows Target maximum number of rows
     * @param cols Target maximum number of columns
     */
    constexpr void resize(size_t rows, size_t cols) {
        if (dist.size() < cols) {
            dist.resize(cols);
            par.resize(cols);
            col_in_use.resize(cols, false);
        }
        rows2use.reserve(rows);
        cols2use.reserve(cols);
        heap.reserve(cols);
    }
};


/**
 * @brief Fully solves an assignment problem using SSP
 *
 * Completely re-initializes the Subproblem, ignoring any prior data or bans.
 * Solves the problem from scratch by iteratively matching unmatched rows to
 * columns via shortest augmenting paths
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indexing
 * @tparam MatrixT Type of the cost matrix
 *
 * @param sol Subproblem to be filled with the matching
 * @param C The cost matrix
 * @param W Pre-allocated worker buffers
 * @param max_cost Maximum allowed cost boundary
 * @param base_cost Base cost of a subproblem - it's added to the cost
 *
 * @return The total cost of the Subproblem
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
Scalar full_ssp(
    Subproblem<Scalar, Idx>& sol, const MatrixT& C, SSPWorkers<Scalar, Idx>& W, Scalar max_cost = std::numeric_limits<Scalar>::max(), Scalar base_cost = 0);


/**
 * @brief Solves a partially solved subproblem by re-matching the last row
 *
 * Assumes all rows (including the last one) and all columns are currently matched. Re-matches
 * the last row while respecting bans for the last row. Uses SSP to find the
 * best augmenting path. Fails if the cost change exceeds max_cost
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indexes
 * @tparam MatrixT Type of the cost matrix
 *
 * @param sol Partially solved Subproblem with all rows and cols matched
 * @param C The cost matrix
 * @param W Pre-allocated worker buffers
 * @param max_cost Maximum allowed increase in cost
 *
 * @return The cost change for re-matching the last row, or maximum value if impossible
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
Scalar ssp_last(Subproblem<Scalar, Idx>& sol, const MatrixT& C, SSPWorkers<Scalar, Idx>& W, Scalar max_cost = std::numeric_limits<Scalar>::max());

//==========================================================================================

namespace internal {

/**
 * @brief Context for active SSP computations
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indexes
 */
template <typename Scalar, typename Idx>
struct SSPContext {
    std::span<const Idx> cols2use;       ///< Active columns remaining to explore
    Scalar cost;                   ///< Current cost in SSP
    Scalar miss_cost;              ///< Cost to reach missing zone
    Scalar miss_cost_past;         ///< Cost to missing zone before it was banned
    Idx miss_par;                  ///< Parent in path to missing zone
    Idx sink;                      ///< Current sink column in augmenting path
    bool miss_from_row;            ///< Whether missing zone was explored from row
    bool allow_miss;               ///< Whether missing zone wasnt explored
};

/**
 * @brief Initializes the SSP by matchings rows with shortest augmented path of length 1
 *
 * Creates an initial partial matching where each row is matched to its
 * cheapest available column. Rows that cannot be matched (because their
 * best column is already taken) are added to the work queue
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indexes
 * @tparam MatrixT Type of the cost matrix
 *
 * @param sol Subproblem being initialized
 * @param C The cost matrix
 * @param W Worker buffers
 *
 * @return The total cost of the initial greedy matching
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
Scalar ssp_initialization(Subproblem<Scalar, Idx>& sol, const MatrixT& C, SSPWorkers<Scalar, Idx>& W);

/**
 * @brief Finds the minimum distance column among unclosed columns
 *
 * Searches through active columns and returns the one with minimum distance,
 * comparing against an initial cost/index pair
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indexes
 *
 * @param cols2use Span of active column indices
 * @param dist Distances to columns
 * @param init_cost Initial comparison cost
 * @param init_idx Initial comparison index
 *
 * @return Pair of minimum distance and index in cols2use span
 */
template <typename Scalar, typename Idx>
std::pair<Scalar, Idx> find_min_col(
        std::span<const Idx> cols2use, const std::vector<Scalar>& dist,
        Scalar init_cost = std::numeric_limits<Scalar>::max(), Idx init_idx = -Idx(1));

/**
 * @brief Finds the minimum distance column using a binary heap
 *
 * Extradicts the smallest element from the heap while skipping invalid or already used columns
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indexes
 *
 * @param h Binary heap containing available columns and their costs
 * @param col_in_use Flags indicating whether a column is valid to process
 * @param EMPTY Constant representing an empty matching state
 *
 * @return Pair of minimum distance and column index
 */
template <typename Scalar, typename Idx>
std::pair<Scalar, Idx> find_min_col(BinaryHeap<QElem<Scalar, Idx>>& h, const std::vector<uint8_t>& col_in_use, Idx EMPTY);

/**
 * @brief Expands Dijkstra from given row in a dense matrix
 *
 * Updates distances to all columns reachable from the given row.
 * Also updates the best path to the augmenting column
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indexes
 *
 * @param row Row index
 * @param matched_col Column index that the row is matched to
 * @param sol The current subproblem
 * @param cx Context containing current state
 * @param C The dense cost matrix
 * @param W Worker buffers
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT>)
void dijkstra_row_expansion(
        Idx row, Idx matched_col, const Subproblem<Scalar, Idx>& sol, internal::SSPContext<Scalar, Idx>& cx,
        const MatrixT& C, SSPWorkers<Scalar, Idx>& W);

/**
 * @brief Expands Dijkstra from given row in a sparse matrix
 *
 * Updates distances to all connected columns reachable from the given row.
 * Also updates the best path to the augmenting column using heap insertions
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indexes
 *
 * @param row Row index
 * @param matched_col Column index that the row is matched to
 * @param sol The current subproblem
 * @param cx Context containing current state
 * @param C The sparse cost matrix
 * @param W Worker buffers
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT>)
void dijkstra_row_expansion(
        Idx row, Idx matched_col, const Subproblem<Scalar, Idx>& sol, internal::SSPContext<Scalar, Idx>& cx,
        const MatrixT& C, SSPWorkers<Scalar, Idx>& W);

/**
 * @brief Expands Dijkstra's missing zone
 *
 * Expands the missing zone by updating distances to all columns
 * and reaching all rows matched to missing zone and expanding them
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indeces
 * @tparam MatrixT Type of the cost matrix
 *
 * @param matched_col Column index currently matched
 * @param sol The current Subproblem
 * @param cx Context containing current state
 * @param C The cost matrix
 * @param W Worker buffers
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
void dijkstra_expand_miss_zone(
        Idx matched_col, const Subproblem<Scalar, Idx>& sol, internal::SSPContext<Scalar, Idx>& cx,
        const MatrixT& C, SSPWorkers<Scalar, Idx>& W);

/**
 * @brief Matches the given row assuming no augmenting rows are matched
 *
 * Uses Dijkstra's algorithm to find the shortest augmenting path from the given
 * row to an unmatched column (normal or augmenting). Assumes that the given
 * row is unmatched and all augmenting rows are unmatched
 *
 * @tparam Scalar Type used in cost matrix
 * @tparam Idx Type used for indexes
 * @tparam MatrixT Type of the cost matrix
 *
 * @param row Index of the unmatched row
 * @param sol Subproblem being updated with new matching
 * @param C The cost matrix
 * @param W Worker buffers
 * @param max_cost Maximum allowed cost limit
 *
 * @return The cost of the augmenting path
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
Scalar dijkstra_full(
        Idx row_idx, Subproblem<Scalar, Idx>& sol, const MatrixT& C,
        SSPWorkers<Scalar, Idx>& W, Scalar max_cost = std::numeric_limits<Scalar>::max());

/**
 * @brief Sets a specific value for a range of elements in a vector
 *
 * @tparam T Type of the destination vector elements
 * @tparam Container Type of the indices container
 * @tparam V Type of the value to set
 *
 * @param dst Destination vector
 * @param idxs Container specifying the indices to update
 * @param val Value to assign
 * @param dst_size Optional logical size of the destination boundary
 */
template <typename T, typename Container, typename V>
void set_range(std::vector<T>& dst, const Container& idxs, V val, size_t dst_size = std::numeric_limits<size_t>::max()) {
    if (dst_size == std::numeric_limits<size_t>::max()) dst_size = dst.size();
    assert(dst_size <= dst.size());

    constexpr size_t SPARSE_THR = 64 / sizeof(T);
    if (idxs.size() * SPARSE_THR < dst_size) {
        for (auto idx : idxs) {
            dst[idx] = static_cast<T>(val);
        }
    } else {
        std::fill_n(dst.begin(), dst_size, static_cast<T>(val));
    }
}

} // namespace internal

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
Scalar full_ssp(Subproblem<Scalar, Idx>& sol, const MatrixT& C, SSPWorkers<Scalar, Idx>& W, Scalar max_cost, Scalar base_cost) {
    using std::swap;
    using SolT = Subproblem<Scalar, Idx>;
    constexpr bool IS_DENSE = is_matrix_v<MatrixT>;
    constexpr Scalar INF = std::numeric_limits<Scalar>::max();

    // For now, max_cost isnt working properly because of negative costs in the matrix
    max_cost = INF;

    sol.prepare(C.rows(), C.cols());
    sol.cur_cost = base_cost;

    #ifndef NDEBUG
    for (auto row : sol.rows2use) assert(row < static_cast<Idx>(sol.max_rows()));
    for (auto col : sol.cols2use) assert(col < static_cast<Idx>(sol.max_cols()));
    if constexpr (!IS_DENSE) {
        for (size_t row = 0; row < C.rows(); row++) {
            auto elems = C.row_elems(row);
            for (const auto& elem : elems) assert (elem.col < static_cast<Idx>(sol.max_cols()));
        }
    }
    #endif

    if (C.cols() == 0) {
        for (auto row : sol.rows2use) {
            sol.col4row[row] = SolT::EMPTY;
        }
    }
    if (C.rows() == 0) {
        for (auto col : sol.cols2use) {
            sol.row4col[col] = SolT::EMPTY;
        }
    }
    if (C.rows() == 0 || C.cols() == 0)
        return base_cost < max_cost ? base_cost : INF;

    W.rows2use.clear();
    W.cols2use.clear();
    W.resize(sol.max_rows(), sol.max_cols());
    #ifndef NDEBUG
    for (auto col : W.col_in_use) assert(!col && "W.col_in_use must be initialized to all zeros");
    #endif
    if constexpr (IS_DENSE) {
        W.cols2use = sol.cols2use;
    } else {
        for (auto col : sol.cols2use) {
            W.col_in_use[col] = true;
        }
    }

    if (internal::ssp_initialization(sol, C, W) >= max_cost) {
        if constexpr (!IS_DENSE) internal::set_range(W.col_in_use, sol.cols2use, 0, sol.max_cols());
        #ifndef NDEBUG
        for (auto col : W.col_in_use) assert(!col && "W.col_in_use wasnt fully reseted");
        #endif
        return INF;
    }
    max_cost -= sol.cur_cost;

    for (auto row : W.rows2use) {
        Scalar cost = internal::dijkstra_full(row, sol, C, W, max_cost);
        if (cost >= max_cost) {
            if constexpr (!IS_DENSE) internal::set_range(W.col_in_use, sol.cols2use, 0, sol.max_cols());
            #ifndef NDEBUG
            for (auto col : W.col_in_use) assert(!col && "W.col_in_use wasnt fully reseted");
            #endif
            return INF;
        }
        max_cost -= cost;
    }
    // set the rest of the columns as matched to augmenting row
    for (auto col : sol.cols2use) {
        if (sol.row4col[col] == SolT::UNMATCHED)
            sol.row4col[col] = SolT::EMPTY;
    }
    if constexpr (!IS_DENSE) internal::set_range(W.col_in_use, sol.cols2use, 0, sol.max_cols());
    
    #ifndef NDEBUG
    validate_sol(sol, C);
    for (auto col : W.col_in_use) assert(!col && "W.col_in_use wasnt fully reseted");
    #endif

    return sol.cur_cost < max_cost ? sol.cur_cost : INF;
}

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
Scalar ssp_last(Subproblem<Scalar, Idx>& sol, const MatrixT& C, SSPWorkers<Scalar, Idx>& W, Scalar max_cost) {
    using SolT = Subproblem<Scalar, Idx>;
    constexpr bool IS_DENSE = is_matrix_v<MatrixT>;
    const Scalar INF = std::numeric_limits<Scalar>::max();

    W.rows2use.clear();
    W.cols2use.clear();
    W.resize(sol.max_rows(), sol.max_cols());

    // the curently unassigned row is the one at the end
    Idx row = sol.rows2use.back();
    Idx matched_col = sol.col4row[row];
    assert(matched_col != SolT::UNMATCHED);

    Scalar u = 0;
    if constexpr (IS_DENSE) {
        W.cols2use = sol.cols2use;
        if (matched_col != SolT::EMPTY)
            u = C(row, matched_col) - sol.v[matched_col];
        // setting up initial distances
        for (auto col : sol.cols2use) {
            if (sol.banned_cols[col]) {
                W.dist[col] = INF;
                W.par[col] = SolT::UNMATCHED;
            } else {
                W.dist[col] = C(row, col) - sol.v[col] - u;
                W.par[col] = row;
            }
        }
    } else {
        #ifndef NDEBUG
        for (auto col : W.col_in_use) assert(!col && "W.col_in_use isnt initalized properly");
        #endif
        W.cols2use.clear();
        auto elems = C.row_elems(row);
        if (matched_col != SolT::EMPTY) {
            auto it = elems.size() >= LB_THR ? std::ranges::lower_bound(elems, matched_col, std::less<Idx>{}, &MatrixT::Elem::col)
                                             : std::ranges::find(elems, matched_col, &MatrixT::Elem::col);
            assert(it != elems.end() && it->col == matched_col && "Matched to unconnected column");
            u = it->val - sol.v[matched_col];
        }
        for (auto col : sol.cols2use) {
            W.col_in_use[col] = true;
        }
        internal::set_range(W.dist, sol.cols2use, max_cost, sol.max_cols());
        W.heap.clear();
        for (size_t j = 0; j < elems.size(); j++) {
            auto col = elems[j].col;
            Scalar val = elems[j].val - sol.v[col] - u;
            if (W.col_in_use[col] && val < max_cost && !sol.banned_cols[col]) {
                W.dist[col] = val;
                W.par[col] = row;
                W.heap.push({val, col});
            }
        }
    }

    // storing the variables in a context struct
    internal::SSPContext<Scalar, Idx> cx{
        .cols2use = std::span<const Idx>(W.cols2use.data(), sol.cols()),
        .cost = Scalar(0),
        .miss_cost = INF,
        .miss_cost_past = INF,
        .miss_par = SolT::UNMATCHED,
        .sink = SolT::UNMATCHED,
        .miss_from_row = true,
        .allow_miss = true
    };
    if (sol.allow_miss) {
        if constexpr (!IS_DENSE) W.heap.push({-u, SolT::EMPTY});
        cx.miss_cost = -u;
        cx.miss_par = row;
    }

    // The main loop, running dijkstra until we get to the
    // column that was matched to initial row before
    while (cx.sink != matched_col) {
        auto [min_cost, min_col] = [&](){
            if constexpr (IS_DENSE) return internal::find_min_col(cx.cols2use, W.dist, cx.miss_cost, SolT::EMPTY);
            else return internal::find_min_col(W.heap, W.col_in_use, SolT::EMPTY);
        }();
        cx.cost = min_cost;
        assert(min_cost > -EPS && "Negative slack/cost");
        // INF cost or already worse than the top k assignments found
        if (min_cost >= max_cost) {
            if constexpr (!IS_DENSE) internal::set_range(W.col_in_use, sol.cols2use, 0, sol.max_cols());
            #ifndef NDEBUG
            for (auto col : W.col_in_use) assert(!col && "W.col_in_use wasnt fully reseted");
            #endif
            return INF;
        }
        if (min_col == SolT::EMPTY) {
            if (!cx.allow_miss) continue;
            internal::dijkstra_expand_miss_zone(matched_col, sol, cx, C, W);
            continue;
        }
        cx.sink = min_col;
        if constexpr (IS_DENSE)
            cx.sink = cx.cols2use[min_col];
        // found the non-matched column
        if (cx.sink == matched_col) break;

        if constexpr (IS_DENSE) {
            // set column is closed for dijsktra - remove it from cols2use
            std::swap(W.cols2use[min_col], W.cols2use[cx.cols2use.size() - 1]);
            cx.cols2use = cx.cols2use.first(cx.cols2use.size() - 1);
        } else {
            W.cols2use.push_back(min_col);
            W.col_in_use[min_col] = false;
        }

        row = sol.row4col[cx.sink];
        if (row == SolT::EMPTY) {
            // missing zone was already expanded
            if (!cx.allow_miss) continue;
            cx.miss_par = cx.sink;
            cx.miss_from_row = false;
            internal::dijkstra_expand_miss_zone(matched_col, sol, cx, C, W);
        // Update distances as normal
        } else {
            internal::dijkstra_row_expansion(row, cx.sink, sol, cx, C, W);
        }
    }

    // augment the path
    do {
        // we got to empty from:
        // - row: this row should now be matched empty
        // - col: just go to that col
        if (cx.sink != SolT::EMPTY) {
            row = W.par[cx.sink];
            sol.row4col[cx.sink] = row;
        }
        if (cx.sink == SolT::EMPTY || row == SolT::EMPTY) {
            if (!cx.miss_from_row) {
                cx.sink = cx.miss_par;
            } else {
                row = cx.miss_par;
                assert(row != SolT::UNMATCHED && row != SolT::EMPTY);
                cx.sink = sol.col4row[row];
                sol.col4row[row] = SolT::EMPTY;
            }
        } else {
            std::swap(sol.col4row[row], cx.sink);
        }
    } while (row != sol.rows2use.back());

    // update of reductions
    // if missing zone was not reached, we update as normal
    // if it was, we need to alter the v's by adding cost - miss_cost to them
    // this wil transform update as v_j = (v_j - cost + dist_j) + (cost - cost) = v_j + dist_j - miss_cost
    if (!cx.allow_miss) {
        for (auto col : sol.cols2use) {
            sol.v[col] += cx.cost - cx.miss_cost_past;
        }
    }
    // update the reductions
    if constexpr (IS_DENSE) {
        // conveniently, all closed columns are at the end of W.cols2use, right behind the span
        size_t closed = W.cols2use.size() - cx.cols2use.size();
        cx.cols2use = std::span<const Idx>(W.cols2use.data() + cx.cols2use.size(), closed);
    } else {
        cx.cols2use = std::span<const Idx>(W.cols2use.data(), W.cols2use.size());
        internal::set_range(W.col_in_use, sol.cols2use, 0, sol.max_cols());
    }
    for (auto col : cx.cols2use) {
        sol.v[col] += -cx.cost + W.dist[col];
    }

    // change the cost inside sol
    sol.cur_cost += cx.cost;

    #ifndef NDEBUG
    validate_sol(sol, C);
    for (auto col : W.col_in_use) assert(!col && "W.col_in_use wasnt fully reseted");
    #endif

    return cx.cost;
}

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
Scalar internal::ssp_initialization(Subproblem<Scalar, Idx>& sol, const MatrixT& C, SSPWorkers<Scalar, Idx>& W) {
    constexpr bool IS_DENSE = is_matrix_v<MatrixT>;
    using SolT = Subproblem<Scalar, Idx>;

    Scalar cost = 0;

    for (Idx row : std::views::reverse(sol.rows2use)) {
        Scalar min_slack = 0; // 0 to augmenting col
        Idx min_col = SolT::EMPTY;

        // Finding minimum cost column
        if constexpr (IS_DENSE) {
            for (auto col : sol.cols2use) {
                Scalar slack = C(row, col);
                if (slack < min_slack) {
                    min_slack = slack;
                    min_col = col;
                }
            }
        } else {
            auto elems = C.row_elems(row);
            for (size_t j = 0; j < elems.size(); j++) {
                auto col = elems[j].col;
                if (!W.col_in_use[col]) continue;
                Scalar slack = elems[j].val;
                if (slack < min_slack) {
                    min_slack = slack;
                    min_col = col;
                }
            }
        }
        
        // row can be matched to unmatched column, its augmenting column is also unmatched
        if (min_col == SolT::EMPTY || sol.row4col[min_col] == SolT::UNMATCHED) {
            sol.col4row[row] = min_col;
            if (min_col != SolT::EMPTY) {
                sol.row4col[min_col] = row;
                cost += min_slack;
            }
        // row unmatched, push it to rows2use
        } else {
            W.rows2use.push_back(row);
        }
    }
    sol.cur_cost += cost;
    return cost;
}

//==========================================================================================

template <typename Scalar, typename Idx>
std::pair<Scalar, Idx> internal::find_min_col(
        std::span<const Idx> cols2use, const std::vector<Scalar>& dist, Scalar init_cost, Idx init_idx) {

    for (size_t j = 0; j < cols2use.size(); j++) {
        Idx col = cols2use[j];
        if (dist[col] < init_cost) {
            init_cost = dist[col];
            init_idx = j;
        }
    }
    return {init_cost, init_idx};
}

template <typename Scalar, typename Idx>
std::pair<Scalar, Idx> internal::find_min_col(
        BinaryHeap<typename internal::QElem<Scalar, Idx>>& h, const std::vector<uint8_t>& col_in_use, Idx EMPTY) {

    while (!h.empty()) {
        auto elem = h.min();
        h.pop();
        // curent element is the best one
        if (elem.col == EMPTY || col_in_use[elem.col])
            return {elem.cost, elem.col};
    }
    return {std::numeric_limits<Scalar>::max(), -Idx(1)};
}

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT>)
void internal::dijkstra_row_expansion(
        Idx row, Idx matched_col, const Subproblem<Scalar, Idx>& sol, internal::SSPContext<Scalar, Idx>& cx,
        const MatrixT& C, SSPWorkers<Scalar, Idx>& W) {
    
    Scalar u = C(row, matched_col) - sol.v[matched_col];
    const Scalar update_cost = -u + cx.cost;
    // update shortest path to aug. columns
    if (cx.allow_miss && update_cost < cx.miss_cost) {
        cx.miss_cost = update_cost;
        cx.miss_par = row;
    }
    // expand from this row
    for (auto col : cx.cols2use) {
        Scalar val = C(row, col) - sol.v[col] + update_cost;
        if (val < W.dist[col]) {
            W.dist[col] = val;
            W.par[col] = row;
        }
    }
}

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT>)
void internal::dijkstra_row_expansion(
        Idx row, Idx matched_col, const Subproblem<Scalar, Idx>& sol, internal::SSPContext<Scalar, Idx>& cx,
        const MatrixT& C, SSPWorkers<Scalar, Idx>& W) {

    using SolT = Subproblem<Scalar, Idx>;
    auto elems = C.row_elems(row);

    auto it = elems.size() >= LB_THR ? std::ranges::lower_bound(elems, matched_col, std::less<Idx>{}, &MatrixT::Elem::col)
                                     : std::ranges::find(elems, matched_col, &MatrixT::Elem::col);
    assert(it != elems.end() && it->col == matched_col && "Matched to unconnected column");
    Scalar u = it->val - sol.v[matched_col];

    const Scalar update_cost = -u + cx.cost;
    // update shortest path to aug. columns
    if (cx.allow_miss && update_cost < cx.miss_cost) {
        cx.miss_cost = update_cost;
        cx.miss_par = row;
        W.heap.push({update_cost, SolT::EMPTY});
    }
    // expand from this row
    for (size_t j = 0; j < elems.size(); j++) {
        Idx col = elems[j].col;
        if (!W.col_in_use[col]) continue;
        Scalar val = elems[j].val - sol.v[col] + update_cost;
        if (val < W.dist[col]) {
            W.dist[col] = val;
            W.par[col] = row;
            W.heap.push({val, col});
        }
    }
}

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
void internal::dijkstra_expand_miss_zone(
        Idx matched_col, const Subproblem<Scalar, Idx>& sol, internal::SSPContext<Scalar, Idx>& cx,
        const MatrixT& C, SSPWorkers<Scalar, Idx>& W) {

    using SolT = Subproblem<Scalar, Idx>;
    constexpr bool IS_DENSE = is_matrix_v<MatrixT>;
    if (matched_col == SolT::EMPTY) {
        cx.sink = SolT::EMPTY;
        return;
    }
    cx.allow_miss = false;
    // ban miss_cost from becoming the cx.cost column, but store the value
    cx.miss_cost_past = cx.cost;
    cx.miss_cost = std::numeric_limits<Scalar>::max();

    if constexpr (IS_DENSE) {
        // expand to all the rows matched to augmenting column - path there is free
        for (auto row : sol.rows2use) {
            if (sol.col4row[row] != SolT::EMPTY) continue;
            for (auto col : cx.cols2use) {
                // ui is 0 as row is matched to augmenting column
                Scalar val = C(row, col) - sol.v[col] + cx.cost;
                if (val < W.dist[col]) {
                    W.dist[col] = val;
                    W.par[col] = row;
                }
            }
        }
        // exapand to all columns
        for (auto col : cx.cols2use) {
            Scalar val = -sol.v[col] + cx.cost;
            if (val < W.dist[col]) {
                W.dist[col] = val;
                W.par[col] = SolT::EMPTY;
            }
        }
    } else {
        // expand to all the rows matched to augmenting column - path there is free
        for (auto row : sol.rows2use) {
            if (sol.col4row[row] != SolT::EMPTY) continue;
            auto elems = C.row_elems(row);
            for (size_t j = 0; j < elems.size(); j++) {
                Idx col = elems[j].col;
                if (!W.col_in_use[col]) continue;
                Scalar val = elems[j].val - sol.v[col] + cx.cost;
                if (val < W.dist[col]) {
                    W.dist[col] = val;
                    W.par[col] = row;
                    W.heap.push({val, col});
                }
            }
        }
        // exapand to all columns
        for (auto col : sol.cols2use) {
            if (!W.col_in_use[col]) continue;
            Scalar val = -sol.v[col] + cx.cost;
            if (val < W.dist[col]) {
                W.dist[col] = val;
                W.par[col] = SolT::EMPTY;
                W.heap.push({val, col});
            }
        }
    }
}

//==========================================================================================

template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT> || is_matrix_v<MatrixT>)
Scalar internal::dijkstra_full(
        Idx row, Subproblem<Scalar, Idx>& sol, const MatrixT& C, SSPWorkers<Scalar, Idx>& W, Scalar max_cost) {

    using SolT = Subproblem<Scalar, Idx>;
    constexpr bool IS_DENSE = is_matrix_v<MatrixT>;

    assert(sol.col4row[row] == SolT::UNMATCHED);

    auto reset_colinuse = [&](){
        for (auto col : W.cols2use) {
            W.col_in_use[col] = true;
        }
    };

    if constexpr (IS_DENSE) {
        // initialize distances from curent row
        for (auto col : sol.cols2use) {
            W.dist[col] = C(row, col) - sol.v[col];
            W.par[col] = row;
        }
    } else {
        W.cols2use.clear();
        internal::set_range(W.dist, sol.cols2use, max_cost, sol.max_cols());
        auto elems = C.row_elems(row);
        W.heap.clear();
        W.heap.push({0, SolT::EMPTY});
        for (size_t j = 0; j < elems.size(); j++) {
            auto col = elems[j].col;
            Scalar val = elems[j].val - sol.v[col];
            if (W.col_in_use[col] && val < max_cost) {
                W.dist[col] = val;
                W.par[col] = row;
                W.heap.push({val, col});
            }
        }
    }
    // reduction u for the row i should be initialized to min(C_ij) accross all j
    // to make the reduced cost matrix non-negative if there are some negative values
    // in the original cost matrix. However, we can initialize it implicitely. This
    // can make the paths found via dijkstra negative, but since it is a constant added to
    // all the paths from row i, dijkstra will still work and even the reduction updates will
    // work as intended, because the contant term will zero out for all the reductions except
    // for the initial row, which is exactly what we want
    internal::SSPContext<Scalar, Idx> cx{
        .cols2use = std::span<const Idx>(W.cols2use.data(), W.cols2use.size()),
        .cost = Scalar(0),
        .miss_cost = Scalar(0),
        .miss_cost_past = Scalar(0), // unused
        .miss_par = row,
        .sink = SolT::UNMATCHED,
        .miss_from_row = true, // unused
        .allow_miss = true
    };

    while (true) {
        auto [min_cost, min_col] = [&](){
            if constexpr (IS_DENSE) return internal::find_min_col(cx.cols2use, W.dist, cx.miss_cost, SolT::EMPTY);
            else return internal::find_min_col(W.heap, W.col_in_use, SolT::EMPTY);
        }();
        if (min_cost >= max_cost) {
            if constexpr (!IS_DENSE) reset_colinuse();
            return std::numeric_limits<Scalar>::max();
        }
        cx.sink = min_col;
        cx.cost = min_cost;
        if (min_col == SolT::EMPTY) break;

        if constexpr (IS_DENSE)
            cx.sink = cx.cols2use[min_col];

        Idx row = sol.row4col[cx.sink];
        assert(row != SolT::EMPTY);
        if (row == SolT::UNMATCHED) break;

        if constexpr (IS_DENSE) {
            // set column is closed for dijsktra - remove it from cols2use
            std::swap(W.cols2use[min_col], W.cols2use[cx.cols2use.size() - 1]);
            cx.cols2use = cx.cols2use.first(cx.cols2use.size() - 1);
        } else {
            W.cols2use.push_back(min_col);
            W.col_in_use[min_col] = false;
        }
        internal::dijkstra_row_expansion(row, cx.sink, sol, cx, C, W);
    }
    assert(cx.sink != SolT::UNMATCHED);
    // update the matching
    if (cx.sink == SolT::EMPTY) {
        cx.sink = sol.col4row[cx.miss_par];
        sol.col4row[cx.miss_par] = SolT::EMPTY;
    }
    while (cx.sink != SolT::UNMATCHED) {
        Idx row = W.par[cx.sink];
        sol.row4col[cx.sink] = row;
        std::swap(sol.col4row[row], cx.sink);
    }

    // update the reductions
    if constexpr (IS_DENSE) {
        // conveniently, all closed columns are at the end of W.cols2use, right behind the span
        size_t closed = W.cols2use.size() - cx.cols2use.size();
        cx.cols2use = std::span<const Idx>(W.cols2use.data() + cx.cols2use.size(), closed);
    } else {
        cx.cols2use = std::span<const Idx>(W.cols2use.data(), W.cols2use.size());
        reset_colinuse();
    }
    for (auto col : cx.cols2use) {
        sol.v[col] += -cx.cost + W.dist[col];
    }

    // updating sol cost
    sol.cur_cost += cx.cost;
    return cx.cost;
}

} // namspace murty