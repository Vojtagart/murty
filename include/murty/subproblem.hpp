/**
 * @file      subproblem.hpp
 * @brief     Defines subproblem struct for the murty's algortihm
 * @author    @vojtagart
 * @date      21/02/2026
 * @see       https://github.com/Vojtagart/murty
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini.
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "matrix.hpp"
#include "sparse_matrix.hpp"


namespace murty {

constexpr double EPS = 1e-5;

/**
 * @brief State of a matching subproblem
 *
 * Maintains active rows and columns, current matching assignments, dual variables,
 * and problem-specific constraints like banned columns
 *
 * @tparam Scalar The value type used for costs and dual variables
 * @tparam Idx The index type used for rows and columns
 */
template <typename Scalar=double, typename Idx=int>
struct Subproblem {

    constexpr static const Idx UNMATCHED = -Idx(1);
    constexpr static const Idx EMPTY = -Idx(2);

    size_t matrix_idx;
    Scalar cur_cost;                            ///< Current cost of (partial) solution stored
    #ifndef NDEBUG
    Scalar base_cost = 0;                       ///< Base cost of the subproblem, used for debugging
    #endif
    bool allow_miss;                            ///< Whether to allow miss for the last row
    
    std::vector<Idx> rows2use, cols2use;        ///< List of rows/cols considered by this subproblem
    std::vector<Scalar> v;                      ///< Column dual variables
    std::vector<Idx> col4row, row4col;          ///< Matched counterparts or EMPTY/UNMATCHED
    std::vector<uint8_t> banned_cols;           ///< Bans applies only to the last row

    /**
     * @brief Initializes a subproblem
     * 
     * @param matrix_idx Identifier for the cost matrix associated with this subproblem
     * @param rows Number of rows in initial problem
     * @param cols Number of columns in initial problem
     * @param allow_miss Whether miss is allowed for the last column
     * @param cur_cost Initial cost of the subproblem
     */
    constexpr Subproblem(size_t matrix_idx, size_t rows, size_t cols, bool allow_miss = true, Scalar cur_cost = 0)
            : matrix_idx(matrix_idx), cur_cost(cur_cost), allow_miss(allow_miss), v(cols, 0),
              col4row(rows, UNMATCHED), row4col(cols, UNMATCHED), banned_cols(cols, false) {
        rows2use.reserve(rows);
        cols2use.reserve(cols);
    }

    /**
     * @return Number of active rows in the subproblem
     */
    [[nodiscard]] constexpr size_t rows() const noexcept {
        return rows2use.size();
    }
    /**
     * @return Number of active columns in the subproblem
     */
    [[nodiscard]] constexpr size_t cols() const noexcept {
        return cols2use.size();
    }
    /**
     * @return Index+1 of the highest row initially in the subproblem
     */
    [[nodiscard]] constexpr size_t max_rows() const noexcept {
        return col4row.size();
    }
    /**
     * @return Index+1 of the highest column initially in the subproblem
     */
    [[nodiscard]] constexpr size_t max_cols() const noexcept {
        return row4col.size();
    }

    /**
     * @brief Resizes underlying containers for a new maximum row/col indexes
     *
     * @param rows New maximum row index+1
     * @param cols New maximum column index+1
     */
    constexpr void resize(size_t rows, size_t cols) {
        col4row.resize(rows);
        row4col.resize(cols);
        v.resize(cols);
        banned_cols.resize(cols);
        rows2use.reserve(rows);
        cols2use.reserve(cols);
    }
    /**
     * @brief Prepares the subproblem state for a new solver run
     *
     * Clears assignments, resets dual variables and banned columns, and restores initial flags
     *
     * @param rows Number of rows to prepare
     * @param cols Number of columns to prepare
     */
    constexpr void prepare(size_t rows, size_t cols) {
        col4row.assign(rows, UNMATCHED);
        row4col.assign(cols, UNMATCHED);
        v.assign(cols, Scalar(0));
        banned_cols.assign(cols, false);
        cur_cost = 0;
        allow_miss = true;
    }
    
    /**
     * @brief Copies the matching assignment and dual variables from another subproblem
     *
     * @param other The subproblem to copy matching state from
     */
    constexpr void copy_matching(const Subproblem& other) {
        std::copy_n(other.row4col.data(), other.row4col.size(), row4col.data());
        std::copy_n(other.col4row.data(), other.col4row.size(), col4row.data());
        std::copy_n(other.v.data(), other.v.size(), v.data());
    }
    
    /**
     * @brief Outputs the subproblem state to a stream
     *
     * @tparam Stream Type of the output stream
     * 
     * @param os Output stream
     * @param sol The subproblem instance to print
     * @return Reference to the output stream
     */
    template <typename Stream>
    friend Stream& operator<< (Stream& os, const Subproblem& sol) {
        os << "\n===================== Subproblem =====================\n";

        auto print_vec = [&]<typename T>(const std::vector<T>& a) {
            for (const auto& x : a) {
                os << x << ' ';
            }
        };
        
        os << "Matrix_idx: " << sol.matrix_idx << '\n';
        os << "Rows2use: "; print_vec(sol.rows2use); os << '\n';
        os << "Cols2use: "; print_vec(sol.cols2use); os << '\n';
        os << "V: "; print_vec(sol.v); os << '\n';
        os << "col4row: "; print_vec(sol.col4row); os << '\n';
        os << "row4col: "; print_vec(sol.row4col); os << '\n';
        os << "BannedCols: ";
        for (size_t i = 0; i < sol.cols(); i++) {
            if (sol.banned_cols[i]) os << i << " ";
        }
        os << '\n';
        os << "Allow_miss: " << (sol.allow_miss ? "Yes" : "No") << '\n';
        os << "Cur cost: " << sol.cur_cost << '\n';

        os << "====================================================\n\n";
        return os;
    }
};


/**
 * @brief Validates a complete subproblem against the cost matrix and optimality conditions
 *
 * Verifies that:
 * - All matchings are consistent and fully specified
 * - Dual variables satisfy complementary slackness
 * - The computed cost matches the stored cost
 * - Column reductions are non-positive (invariant)
 * - Reduction of augmented rows/column and rows/columns matched to them is 0
 *
 * @tparam Scalar Type of values in the cost matrix
 * @tparam Idx Index type used in the subproblem
 * 
 * @param sol The subproblem to validate
 * @param C The cost matrix
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_matrix_v<MatrixT>)
void validate_sol(const Subproblem<Scalar, Idx>& sol, const MatrixT& C) {
    #ifndef NDEBUG
    using SolT = Subproblem<Scalar, Idx>;

    Scalar cost = sol.base_cost;
    std::vector<uint8_t> row_used(sol.max_rows(), false);

    for (auto row : sol.rows2use) {
        row_used[row] = true;
        Idx matched_col = sol.col4row[row];
        assert(matched_col != SolT::UNMATCHED && "Solution isnt fully solved");
        Scalar u = 0;
        if (matched_col != SolT::EMPTY) {
            u = C(row, matched_col) - sol.v[matched_col];
            cost += C(row, matched_col);
            assert(sol.row4col[matched_col] == row && "Matching mismatch");
        }
        for (auto col : sol.cols2use) {
            if (!sol.rows2use.empty() && row == sol.rows2use.back() && sol.banned_cols[col]) continue;
            // check for non-negative slack
            assert(C(row, col) - sol.v[col] - u > -EPS && "Negative slack");
            if (col != matched_col) assert(sol.row4col[col] != row && "Matching mismatch");
        }
    }
    for (auto col : sol.cols2use) {
        assert(sol.v[col] < EPS && "Column reduction is positive (breaking the invariant)");
        assert(sol.row4col[col] != SolT::UNMATCHED && "Solution isnt fully solved");
        // missed columns has reduction 0
        if (sol.row4col[col] == SolT::EMPTY) assert(sol.v[col] > -EPS && "Reduction for missed column isnt 0");
    }
    // Add costs for previopusly fixed rows
    for (size_t row = 0; row < sol.col4row.size(); row++) {
        if (!row_used[row]) {
            Idx col = sol.col4row[row];
            // assert(col != SolT::UNMATCHED && "Solution isnt fully solved");
            if (col != SolT::EMPTY && col != SolT::UNMATCHED)
                cost += C(row, col);
        }
    }
    assert(std::abs(cost - sol.cur_cost) < EPS * std::max(Scalar(1), std::abs(cost))
            && "Cost stored in solution and the actual cost dont match");
    #endif
}

/**
 * @brief Validates a complete subproblem against the sparse cost matrix and optimality conditions
 *
 * Verifies that:
 * - All matchings are consistent and fully specified
 * - Dual variables satisfy complementary slackness
 * - The computed cost matches the stored cost
 * - Column reductions are non-positive (invariant)
 * - Reduction of augmented rows/column and rows/columns matched to them is 0
 *
 * @tparam Scalar Type of values in the cost matrix
 * @tparam Idx Index type used in the subproblem
 * 
 * @param sol The subproblem to validate
 * @param C The sparse cost matrix
 */
template <typename Scalar, typename Idx, typename MatrixT>
requires (is_sparse_matrix_v<MatrixT>)
void validate_sol(const Subproblem<Scalar, Idx>& sol, const MatrixT& C) {
    #ifndef NDEBUG
    using SolT = Subproblem<Scalar, Idx>;

    Scalar cost = sol.base_cost;
    std::vector<uint8_t> row_used(sol.max_rows(), false);
    std::vector<uint8_t> col_in_use(sol.max_cols(), false);
    for (auto col : sol.cols2use) {
        col_in_use[col] = true;
    }

    for (auto row : sol.rows2use) {
        row_used[row] = true;
        Idx matched_col = sol.col4row[row];
        assert(matched_col != SolT::UNMATCHED && "Solution isnt fully solved");
        auto elems = C.row_elems(row);
        Scalar u = 0;
        if (matched_col != SolT::EMPTY) {
            size_t idx = std::lower_bound(elems.begin(), elems.end(), matched_col, [](const auto& elem, Idx col_) { 
                return elem.col < col_; 
            }) - elems.begin();
            assert(idx < elems.size() && elems[idx].col == matched_col && "Matched to unconnected column");
            u = elems[idx].val - sol.v[matched_col];
            cost += elems[idx].val;
        }
        for (size_t j = 0; j < elems.size(); j++) {
            Idx col = elems[j].col;
            Scalar val = elems[j].val;
            if (!col_in_use[col]) continue;
            if (!sol.rows2use.empty() && row == sol.rows2use.back() && sol.banned_cols[col]) continue;
            // check for non-negative slack
            assert(val - sol.v[col] - u > -EPS && "Negative slack");
            if (col != matched_col) assert(sol.row4col[col] != row && "Matching mismatch");
        }
    }
    for (auto col : sol.cols2use) {
        assert(sol.v[col] < EPS && "Column reduction is positive (breaking the invariant)");
        assert(sol.row4col[col] != SolT::UNMATCHED && "Solution isnt fully solved");
        // missed columns has reduction 0
        if (sol.row4col[col] == SolT::EMPTY) assert(sol.v[col] > -EPS && "Reduction for missed column isnt 0");
    }
    // Add costs for previopusly fixed rows
    for (size_t row = 0; row < sol.col4row.size(); row++) {
        if (!row_used[row]) {
            Idx col = sol.col4row[row];
            // assert(col != SolT::UNMATCHED && "Solution isnt fully solved");
            if (col != SolT::EMPTY && col != SolT::UNMATCHED) {
                auto elems = C.row_elems(row);
                size_t idx = std::lower_bound(elems.begin(), elems.end(), col, [](const auto& elem, Idx col_) { 
                    return elem.col < col_; 
                }) - elems.begin();
                assert(idx < elems.size() && elems[idx].col == col && "Matched to unconnected column");
                cost += elems[idx].val;
            }
        }
    }
    if (std::abs(cost - sol.cur_cost) >= EPS * std::max(Scalar(1), std::abs(cost)))
        std::cerr << "Stored cost: " << sol.cur_cost << ", Calculated cost: " << cost << std::endl;
    assert(std::abs(cost - sol.cur_cost) < EPS * std::max(Scalar(1), std::abs(cost))
            && "Cost stored in solution and the actual cost dont match");
    #endif
}

} // namespace murty