/**
 * @file      sparse_matrix.hpp
 * @brief     Implemts sparse matrix for graph traversal
 * @author    @vojtagart
 * @date      21/02/2026
 * @see       https://github.com/Vojtagart/murty
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <cassert>
#include <algorithm>
#include <utility>
#include <limits>
#include <numeric>
#include <span>
#include <cstddef>
#include <stdexcept>
#include <functional>

#include "matrix.hpp"


namespace murty {

/**
 * @brief Row-major sparse matrix
 *
 * Stores a sparse representation of a dense matrix with configurable sparsity
 * level. Each row's elements are stored contiguously with their
 * column indices. Designed for construction-once, read-many access patterns
 *
 * @tparam Scalar The value type stored in the sparse matrix
 * @tparam Idx The index type used to store column indices
 */
template <typename Scalar, typename Idx=int>
class SparseMatrix {
public:
    /**
     * @brief Struct representing row element - column it is in and an associated value
     */
    struct Elem {
        Idx col;
        Scalar val;
    };

    constexpr SparseMatrix()
            : _cols(0), _row_idxs(1, 0) {}
    /**
     * @brief Constructs new SparseMatrix
     *
     * Utilizes this->fill_from()
     *
     * @tparam Comparator Comparison functor for element selection
     * @tparam MatrixT Type of the Matrix
     * 
     * @param mat The source matrix view to sparsify
     * @param max_per_row Maximum elements per row
     * @param max_val Maximum value to include in the sparse matrix
     * @param comp Comparator instance
     */
    template <typename Derived, typename Comparator = std::less<Scalar>>
    constexpr SparseMatrix(
            const Matrix<Derived, Scalar>& mat, size_t max_per_row = std::numeric_limits<size_t>::max(),
            const Scalar& max_val = std::numeric_limits<Scalar>::max(), Comparator comp = Comparator{}) {
        fill_from(mat, max_per_row, max_val, comp);
    }
    /**
     * @brief Constructs directly from standard CSR format arrays
     * 
     * @param cols Number of columns
     * @param vals Span of values
     * @param col_idxs Span of column indices for each value
     * @param row_ptrs Span of row pointers (size = rows + 1)
     */
    constexpr SparseMatrix(
            size_t cols, std::span<const Scalar> vals, std::span<const Idx> col_idxs, std::span<const Idx> row_idxs) {
        fill_from(cols, vals, col_idxs, row_idxs);
    }
    /**
     * @brief Constructs from another sparse matrix given valid rows and columns
     * 
     * @param source The original sparse matrix
     * @param row_subset Allowed row indices from the source
     * @param col_subset Allowed column indices from the source
     */
    constexpr SparseMatrix(
            const SparseMatrix& src, std::span<const Idx> row_subset, std::span<const Idx> col_subset) {
        fill_from(src, row_subset, col_subset);
    }

    /**
     * @brief Fills a sparse matrix with a dense matrix
     *
     * Extracts elements from the dense matrix based on a comparator and
     * optional maximum value. Per row, keeps only the elements that satisfy
     * both the comparator criterion and are within the value threshold.
     * * @note The behavior depends on the Comparator:
     * - With `std::less<int>` and max_val=10, element 8 is kept, 12 is not
     * - With `std::greater<int>` and max_val=10, element 12 is kept, 8 is not
     *
     * @tparam Comparator Comparison functor for element selection
     * @param mat The source matrix view to sparsify
     * @param max_per_row Maximum elements per row
     * @param max_val Maximum value to include in the sparse matrix
     * @param comp Comparator instance
     */
    template <typename Derived, typename Comparator = std::less<Scalar>>
    constexpr void fill_from(
            const Matrix<Derived, Scalar>& mat, size_t max_per_row = std::numeric_limits<size_t>::max(),
            const Scalar& max_val = std::numeric_limits<Scalar>::max(), Comparator comp = Comparator{}) {
        clear();
        max_per_row = std::min(max_per_row, mat.cols());
        reserve(mat.rows(), mat.rows() * std::min(max_per_row, size_t(10)));
        _cols = mat.cols();

        std::vector<Idx> order(_cols);
        std::iota(order.begin(), order.end(), Idx(0));

        for (size_t row = 0; row < mat.rows(); row++) {
            // faster than sorting the whole array
            // Elements on the left side are the ones we want to keep
            if (max_per_row != mat.cols()) {
                std::nth_element(order.begin(), order.begin() + max_per_row - 1, order.end(), [&](Idx x, Idx y){
                    return comp(mat(row, x), mat(row, y));
                });
                // sort the indexes we want so that they are in increasing order
                std::sort(order.begin(), order.begin() + max_per_row);
            }
            for (size_t i = 0; i < max_per_row; i++) {
                auto idx = order[i];
                if (!comp(max_val, mat(row, idx)))
                    _elems.emplace_back(idx, mat(row, idx));
            }
            _row_idxs.push_back(_elems.size());
        }
    }
    /**
     * @brief Fill the SparseMatrix directly from standard CSR format arrays
     * 
     * @param cols Number of columns
     * @param vals Span of values
     * @param col_idxs Span of column indices for each value
     * @param row_ptrs Span of row pointers (size = rows + 1)
     */
    constexpr void fill_from(
            size_t cols, std::span<const Scalar> vals, std::span<const Idx> col_idxs, std::span<const Idx> row_idxs) {
        assert(vals.size() == col_idxs.size() && "Value and column indices size mismatch");
        _cols = cols;
        _row_idxs.assign(row_idxs.begin(), row_idxs.end());
        _elems.clear();
        _elems.reserve(vals.size());
        for (size_t i = 0; i < vals.size(); i++) {
            _elems.emplace_back(col_idxs[i], vals[i]);
        }
    }
    /**
     * @brief Fills from another sparse matrix given valid rows and columns
     * 
     * @param source The original sparse matrix
     * @param row_subset Allowed row indices from the source
     * @param col_subset Allowed column indices from the source
     */
    constexpr void fill_from(
            const SparseMatrix& src, std::span<const Idx> row_subset, std::span<const Idx> col_subset) {
        _cols = col_subset.size();
        _elems.clear();
        _row_idxs.assign(1, 0);
        std::vector<Idx> col_map(src.cols(), Idx(-1));
        for (size_t i = 0; i < col_subset.size(); i++) {
            col_map[col_subset[i]] = static_cast<Idx>(i);
        }
        reserve(row_subset.size(), src.nvals());
        for (Idx row : row_subset) {
            for (const auto& elem : src.row_elems(row)) {
                Idx new_col = col_map[elem.col];
                if (new_col != Idx(-1))
                    _elems.emplace_back(new_col, elem.val);
            }
            _row_idxs.push_back(_elems.size());
        }
    }

    /**
     * @return The number of rows
     */
    [[nodiscard]] constexpr size_t rows() const noexcept {
        return _row_idxs.size() - 1;
    }
    /**
     * @return The number of columns
     */
    [[nodiscard]] constexpr size_t cols() const noexcept {
        return _cols;
    }
    /**
     * @return The count of values
     */
    [[nodiscard]] constexpr size_t nvals() const noexcept {
        return _elems.size();
    }
    /**
     * @return Const reference to the row indices vector
     */
    [[nodiscard]] constexpr const std::vector<Idx>& row_idxs() const noexcept {
        return _row_idxs;
    }
    /**
     * @return Const reference to the values vector
     */
    [[nodiscard]] constexpr const std::vector<Elem>& elems() const noexcept {
        return _elems;
    }

    /**
     * @brief Returns a span of elements in given row
     *
     * The returned span contains all elements in the
     * specified row. Elemnt consists of column index and its value
     *
     * @param row The row index
     * @return A span of elemtns for the specified row
     */
    [[nodiscard]] constexpr std::span<const Elem> row_elems(size_t row) const {
        assert(row < rows() && "Row index out of bounds");
        return std::span<const Elem>(_elems.data() + _row_idxs[row], row_len(row));
    }
    /**
     * @brief Returns the number of non-zero elements in a given row
     *
     * @param row The row index
     * @return The number of elements in the specified row
     */
    [[nodiscard]] constexpr size_t row_len(size_t row) const {
        assert(row < rows() && "Row index out of bounds");
        return _row_idxs[row + 1] - _row_idxs[row];
    }

    /**
     * @brief Reserve space in the sparse matrix storage
     * 
     * @param rows Number of rows to be reserved
     * @param nvals Number of values in the matrix to be reserved
     */
    constexpr void reserve(size_t rows, size_t nvals) {
        _row_idxs.reserve(rows + 1);
        _elems.reserve(nvals);
    }
    /**
     * @brief Clears the sparse matrix data
     */
    constexpr void clear() {
        _row_idxs.clear();
        _row_idxs.push_back(0);
        _elems.clear();
        _cols = 0;
    }

    /**
     * @brief Exchanges the contents of two sparse matrices
     *
     * @param other The sparse matrix to swap with
     */
    constexpr void swap(SparseMatrix& other) noexcept {
        using std::swap;
        swap(_cols, other._cols);
        swap(_row_idxs, other._row_idxs);
        swap(_elems, other._elems);
    }
    /**
     * @brief Exchanges the contents of two sparse matrices
     *
     * @param lhs The first sparse matrix
     * @param rhs The second sparse matrix
     */
    friend constexpr void swap(SparseMatrix& lhs, SparseMatrix& rhs) noexcept {
        lhs.swap(rhs);
    }
private:
    size_t _cols;
    std::vector<Idx> _row_idxs;
    std::vector<Elem> _elems;
};

//==========================================================================================

/**
 * @brief View over a subset of rows in SparseMatrix
 * 
 * @tparam Scalar The value type stored
 * @tparam Idx The index type used
 */
template <typename Scalar, typename Idx=int>
class SparseMatrixView {
public:
    using Elem = typename SparseMatrix<Scalar, Idx>::Elem;

    /**
     * @brief Constructs a view over specific rows of a SparseMatrix
     * 
     * @param mat Underlying sparse matrix
     * @param row_idxs Subset of rows to include in the view
     */
    constexpr SparseMatrixView(const SparseMatrix<Scalar, Idx>& mat, std::vector<Idx> row_idxs)
            : _mat(mat), _row_idxs(std::move(row_idxs)) {}

    /**
     * @return Number of rows in the view
     */
    [[nodiscard]] constexpr size_t rows() const noexcept { 
        return _row_idxs.size(); 
    }
    /**
     * @return Number of columns in the view
     */
    [[nodiscard]] constexpr size_t cols() const noexcept { 
        return _mat.cols();
    }
    
    /**
     * @brief Returns a span of elements in given row
     *
     * The returned span contains all elements in the
     * specified row. Elemnt consists of column index and its value
     *
     * @param row The row index
     * @return A span of elemtns for the specified row
     */
    [[nodiscard]] constexpr std::span<const Elem> row_elems(size_t row) const {
        assert(row < rows() && "Row index out of bounds");
        return _mat.row_elems(_row_idxs[row]);
    }
    /**
     * @brief Returns the number of non-zero elements in a given row
     *
     * @param row The row index
     * @return The number of elements in the specified row
     */
    [[nodiscard]] constexpr size_t row_len(size_t row) const {
        assert(row < rows() && "Row index out of bounds");
        return _mat.row_len(_row_idxs[row]);
    }

    /**
     * @return Const reference to the row indices vector
     */
    [[nodiscard]] constexpr const std::vector<Idx>& row_idxs() const noexcept {
        return _row_idxs;
    }
    /**
     * @return R-value reference of the row indices vector
     */
    [[nodiscard]] constexpr std::vector<Idx> take_row_idxs() && noexcept { 
        return std::move(_row_idxs); 
    }

private:
    const SparseMatrix<Scalar, Idx>& _mat;
    std::vector<Idx> _row_idxs;
};

//==========================================================================================

namespace internal {
    template <typename Scalar, typename Idx>
    std::true_type is_sparse_matrix_impl(const SparseMatrix<Scalar, Idx>*);
    template <typename Scalar, typename Idx>
    std::true_type is_sparse_matrix_impl(const SparseMatrixView<Scalar, Idx>*);
    std::false_type is_sparse_matrix_impl(...);
}

/**
 * @brief Type trait to check if a type represents sparse
 */
template <typename T>
constexpr bool is_sparse_matrix_v = decltype(internal::is_sparse_matrix_impl(std::declval<T*>()))::value;

} // namespace murty