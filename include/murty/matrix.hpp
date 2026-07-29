/**
 * @file      matrix.hpp
 * @brief     Implemts Matrices for dense graph traversals
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
#include <cstddef>


namespace murty {

/**
 * @brief Matrix interface
 * 
 * @tparam Derived Underlkying class
 * @tparam Scalar Value type
 */
template <typename Derived, typename Scalar>
class Matrix {
protected:
    ~Matrix() = default;
public:
    using ScalarT = Scalar;

    constexpr Scalar operator()(size_t row, size_t col) const noexcept {
        return static_cast<const Derived*>(this)->operator()(row, col);
    }
    [[nodiscard]] constexpr size_t rows() const noexcept {
        return static_cast<const Derived*>(this)->rows();
    }
    [[nodiscard]] constexpr size_t cols() const noexcept {
        return static_cast<const Derived*>(this)->cols();
    }
};

namespace internal {
    template <typename D, typename S>
    std::true_type is_matrix_impl(const Matrix<D, S>*);
    std::false_type is_matrix_impl(...);
}
template <typename T>
constexpr bool is_matrix_v = decltype(internal::is_matrix_impl(std::declval<T*>()))::value;

//==========================================================================================

template <typename Derived, typename Scalar>
class TransposedView : public Matrix<TransposedView<Derived, Scalar>, Scalar> {
public:
    constexpr explicit TransposedView(const Matrix<Derived, Scalar>& mat) 
            : _mat(static_cast<const Derived&>(mat)) {}

    constexpr Scalar operator()(size_t row, size_t col) const noexcept {
        return _mat(col, row);
    }
    [[nodiscard]] constexpr size_t rows() const noexcept {
        return _mat.cols();
    }
    [[nodiscard]] constexpr size_t cols() const noexcept {
        return _mat.rows();
    }
private:
    const Derived& _mat;
};

template <typename Derived, typename Scalar>
constexpr TransposedView<Derived, Scalar> transpose_view(const Matrix<Derived, Scalar>& mat) {
    return TransposedView<Derived, Scalar>(mat);
}

//==========================================================================================

/**
 * @brief A 2D matrix
 * 
 * @tparam Scalar The value type stored in the matrix
 */
template <typename Scalar>
class DenseMatrix : public Matrix<DenseMatrix<Scalar>, Scalar> {
public:
    /**
     * @brief Constructs a DenseMatrix
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     */
    constexpr DenseMatrix(size_t rows, size_t cols)
            : _data(rows * cols), _rows(rows), _cols(cols) {}

    /**
     * @brief Constructs a DenseMatrix
     * 
     * @param data Pointer to the flatten data (row-major)
     * @param rows Number of rows
     * @param cols Number of columns
     */
    constexpr DenseMatrix(Scalar* data, size_t rows, size_t cols)
            : _data(data, data + rows * cols), _rows(rows), _cols(cols) {}

    /**
     * @brief Constructs from another arbitrarly matrix
     * 
     * @param other Other matrix
     */
    template <typename Derived, typename U>
    constexpr DenseMatrix(const Matrix<Derived, U>& other)
            : _data(other.rows() * other.cols()), _rows(other.rows()), _cols(other.cols()) {
        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < _cols; j++) {
                this->operator()(i, j) = static_cast<Scalar>(other(i, j));
            }
        }
    }

    /**
     * @brief Accesses value at given row and column
     * 
     * @param row Row index
     * @param col Column index
     * @return Reference to value at the specified row and column
     */
    constexpr Scalar& operator() (size_t row, size_t col) noexcept {
        assert(row < _rows && col < _cols && "Matrix index out of bounds");
        return _data[row * _cols + col];
    }
    /**
     * @brief Accesses value at given row and column
     * 
     * @param row Row index
     * @param col Column index
     * @return Value at the specified row and column
     */
    constexpr Scalar operator() (size_t row, size_t col) const noexcept {
        assert(row < _rows && col < _cols && "Matrix index out of bounds");
        return _data[row * _cols + col];
    }

    /**
     * @return Number of rows in the view
     */
    [[nodiscard]] constexpr size_t rows() const noexcept {
        return _rows;
    }
    /**
     * @return Number of columns in the view
     */
    [[nodiscard]] constexpr size_t cols() const noexcept {
        return _cols;
    }
    /**
     * @return Pointer to the data buffer
     */
    [[nodiscard]] constexpr Scalar* data() noexcept {
        return _data.data();
    }
    /**
     * @return Const pointer to the data buffer
     */
    [[nodiscard]] constexpr const Scalar* data() const noexcept {
        return _data.data();
    }

    /**
     * @brief Resize the Matrix
     * 
     * The resize operation keeps the original elements present in the matrix
     * 
     * @param rows New number of rows
     * @param cols New number of columns
     */
    constexpr void resize(size_t rows, size_t cols) {
        _data.resize(rows * cols);
        _rows = rows;
        _cols = cols;
    }
    /**
     * @brief Reserve capacity for the data bufer
     * 
     * @param cap Capacity to be reserved
     */
    constexpr void reserve(size_t cap) {
        _data.reserve(cap);
    }

    /**
     * @brief Exchanges contents with another DenseMatrix
     * 
     * @param other The matrix to swap with
     */
    constexpr void swap(DenseMatrix& other) noexcept {
        using std::swap;
        swap(_data, other._data);
        swap(_rows, other._rows);
        swap(_cols, other._cols);
    }
    /**
     * @brief Exchanges contents of two Matrices
     * 
     * @param lhs The first matrix
     * @param rhs The second matrix
     */
    friend constexpr void swap(DenseMatrix& lhs, DenseMatrix& rhs) noexcept {
        lhs.swap(rhs);
    }
private:
    std::vector<Scalar> _data;
    size_t _rows, _cols;
};

//==========================================================================================

/**
 * @brief A view over a DenseMatrix
 * 
 * @tparam Scalar The value type stored in the matrix
 */
template <typename Scalar>
class DenseMatrixView : public Matrix<DenseMatrixView<Scalar>, Scalar> {
public:
    /**
     * @brief Constructs a DenseMatrixView
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     */
    constexpr DenseMatrixView(const Scalar* data, size_t rows, size_t cols)
            : _data(data), _rows(rows), _cols(cols) {}

    /**
     * @brief Constructs DenseMatrixView from DenseMatrix
     * 
     */
    constexpr DenseMatrixView(const DenseMatrix<Scalar>& mat)
            : _data(mat.data()), _rows(mat.rows()), _cols(mat.cols()) {}
    constexpr DenseMatrixView(const DenseMatrix<Scalar>&&) = delete;

    /**
     * @brief Accesses value at given row and column
     * 
     * @param row Row index
     * @param col Column index
     * @return Value at the specified row and column
     */
    constexpr Scalar operator() (size_t row, size_t col) const noexcept {
        assert(row < _rows && col < _cols && "Matrix index out of bounds");
        return _data[row * _cols + col];
    }

    /**
     * @return Number of rows in the view
     */
    [[nodiscard]] constexpr size_t rows() const noexcept {
        return _rows;
    }
    /**
     * @return Number of columns in the view
     */
    [[nodiscard]] constexpr size_t cols() const noexcept {
        return _cols;
    }
    /**
     * @return Const pointer to the data buffer
     */
    [[nodiscard]] constexpr const Scalar* data() const noexcept {
        return _data;
    }

    /**
     * @brief Exchanges contents with another DenseMatrix
     * 
     * @param other The matrix to swap with
     */
    constexpr void swap(DenseMatrixView& other) noexcept {
        using std::swap;
        swap(_data, other._data);
        swap(_rows, other._rows);
        swap(_cols, other._cols);
    }
    /**
     * @brief Exchanges contents of two Matrices
     * 
     * @param lhs The first matrix
     * @param rhs The second matrix
     */
    friend constexpr void swap(DenseMatrixView& lhs, DenseMatrixView& rhs) noexcept {
        lhs.swap(rhs);
    }
private:
    const Scalar* _data;
    size_t _rows, _cols;
};

//==========================================================================================

/**
 * @brief A 2D matrix view over some existing data
 * 
 * UB if the underlying data are invalidated
 * 
 * @tparam Scalar The value type stored in the matrix
 * @tparam Idx Type used to index rows/cols
 */
template <typename Scalar, typename Idx=int>
class MatrixView : public Matrix<MatrixView<Scalar, Idx>, Scalar> {
public:
    /**
     * @brief Constructs a MatrixView
     * 
     * @param data Underlying row-major data buffer
     * @param data_cols Number of elements between consecutive rows in the underlying buffer
     * @param row_idxs Subset of rows to use
     * @param col_idxs Subset of columns to use
     */
    constexpr MatrixView(const Scalar* data, size_t data_cols, std::vector<Idx> row_idxs, std::vector<Idx> col_idxs)
            : _data(data), _stride(data_cols), _row_idxs(std::move(row_idxs)), _col_idxs(std::move(col_idxs)) {}

    /**
     * @brief Accesses value at given row and column
     * 
     * @param row Row index
     * @param col Column index
     * @return Value at the specified row and column
     */
    constexpr Scalar operator() (size_t row, size_t col) const noexcept {
        assert(row < rows() && col < cols() && "Matrix index out of bounds");
        return _data[_row_idxs[row] * _stride + _col_idxs[col]];
    }
    /**
     * @brief Exchanges contents with another MatrixView
     * 
     * @param other The matrix view to swap with
     */
    constexpr void swap(MatrixView& other) noexcept {
        using std::swap;
        swap(_data, other._data);
        swap(_stride, other._stride);
        swap(_row_idxs, other._row_idxs);
        swap(_col_idxs, other._col_idxs);
    }
    /**
     * @brief Exchanges contents of two MatrixViews
     * 
     * @param lhs The first matrix
     * @param rhs The second matrix
     */
    friend constexpr void swap(MatrixView& lhs, MatrixView& rhs) noexcept {
        lhs.swap(rhs);
    }

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
        return _col_idxs.size();
    }
    /**
     * @return Const pointer to the underlying data buffer
     */
    [[nodiscard]] constexpr const Scalar* data() const noexcept {
        return _data;
    }
    /**
     * @return The stride of the underlying data buffer
     */
    [[nodiscard]] constexpr size_t stride() const noexcept {
        return _stride;
    }
    /**
     * @return Const reference to the row indices vector
     */
    [[nodiscard]] constexpr const std::vector<Idx>& row_idxs() const noexcept {
        return _row_idxs;
    }
    /**
     * @return Const reference to the column indices vector
     */
    [[nodiscard]] constexpr const std::vector<Idx>& col_idxs() const noexcept {
        return _col_idxs;
    }
    /**
     * @return The row indices vector (rvalue)
     */
    [[nodiscard]] constexpr std::vector<Idx> take_row_idxs() && noexcept {
        return std::move(_row_idxs);
    }
    /**
     * @return The column indices vector (rvalue)
     */
    [[nodiscard]] constexpr std::vector<Idx> take_col_idxs() && noexcept {
        return std::move(_col_idxs);
    }
private:
    const Scalar* _data;
    size_t _stride;
    std::vector<Idx> _row_idxs, _col_idxs;
};

} // namespace murty