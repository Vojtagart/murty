/**
 * @file      bindings.hpp
 * @brief     Bindings for Murty's algorithm
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/murty
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <numeric>
#include <utility>
#include <string>
#include <vector>
#include <cstring>

#include "murty/matrix.hpp"
#include "murty/sparse_matrix.hpp"
#include "murty/solve.hpp"
#include "murty/ssp.hpp"

namespace py = pybind11;


template <typename Scalar, typename Idx>
void bind_assignment_workers(py::module& m, const std::string& name) {
    using T = murty::SSPWorkers<Scalar, Idx>;
    py::class_<T> cls(m, name.c_str());
    cls
        .def(py::init<size_t, size_t>(), py::arg("rows") = 0, py::arg("cols") = 0)
        .def("reserve", &T::resize, py::arg("rows"), py::arg("cols"));
}

template <typename Scalar, typename Idx>
void bind_murty_workers(py::module& m, const std::string& name) {
    using T = murty::MurtyWorkers<Scalar, Idx>;
    py::class_<T> cls(m, name.c_str());
    cls
        .def(py::init<size_t, size_t, size_t>(), py::arg("K") = 0, py::arg("rows") = 0, py::arg("cols") = 0)
        .def("reserve", &T::resize, py::arg("K"), py::arg("rows"), py::arg("cols"));
}

template <typename Scalar, typename Idx>
void bind_assignment_struct(py::module& m, const std::string& name) {
    using T = murty::Assignment<Scalar, Idx>;
    py::class_<T> cls(m, name.c_str());
    cls
        .def(py::init())
        .def_property_readonly("ass", 
            [](T& self) -> py::array_t<Idx> {
                return py::array_t<Idx>(
                    {self.ass.size(), static_cast<size_t>(2)},
                    {sizeof(std::pair<Idx, Idx>), sizeof(Idx)},
                    reinterpret_cast<const Idx*>(self.ass.data()), 
                    py::cast(self)
                );
            }
        )
        .def_readonly("cost", &T::cost);
}

template <typename Scalar, typename Idx>
void bind_sparse_matrix(py::module& m, const std::string& name) {
    using T = murty::SparseMatrix<Scalar, Idx>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    py::class_<T> cls(m, name.c_str());
    cls
        .def(py::init([](const ArrT& mat, size_t max_per_row, Scalar max_val) {
            auto info = mat.request();
            if (info.ndim != 2)
                throw std::invalid_argument("mat must be a 2D matrix");
            size_t rows = info.shape[0];
            size_t cols = info.shape[1];
            murty::DenseMatrixView<Scalar> cmat(static_cast<const Scalar*>(info.ptr), rows, cols);
            return new T(cmat, max_per_row, max_val);

    }), py::arg("mat"), py::arg("max_per_row") = std::numeric_limits<size_t>::max(), py::arg("max_val") = std::numeric_limits<Scalar>::max());
}

template <typename Scalar, typename Idx>
void bind_assignment(py::module& m, const std::string& name) {
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    m.def(name.c_str(), [](const ArrT& C, murty::SSPWorkers<Scalar, Idx>* W, Scalar max_cost) {
        auto info = C.request();
        if (info.ndim != 2)
            throw std::invalid_argument("C must be a 2D matrix");
            
        size_t rows = info.shape[0];
        size_t cols = info.shape[1];
        
        murty::DenseMatrixView<Scalar> mat(static_cast<const Scalar*>(info.ptr), rows, cols);
        murty::Assignment<Scalar, Idx> ass;

        {
            py::gil_scoped_release release;
            if (W) ass = murty::assignment<Scalar, Idx, murty::DenseMatrixView<Scalar>>(mat, *W, max_cost);
            else ass = murty::assignment<Scalar, Idx, murty::DenseMatrixView<Scalar>>(mat, max_cost);
        }

        return ass;

    }, "Solves assignment problem", 
       py::arg("C"), py::arg("workers") = py::none(),
       py::arg("max_cost") = std::numeric_limits<Scalar>::max());
}

template <typename Scalar, typename Idx>
void bind_assignment_sparse(py::module& m, const std::string& name) {
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    m.def(name.c_str(), [](const ArrT& C, size_t max_per_row, Scalar max_val, murty::SSPWorkers<Scalar, Idx>* W, Scalar max_cost) {
        auto info = C.request();
        if (info.ndim != 2)
            throw std::invalid_argument("C must be a 2D matrix");
            
        size_t rows = info.shape[0];
        size_t cols = info.shape[1];
        
        murty::DenseMatrixView<Scalar> mat(static_cast<const Scalar*>(info.ptr), rows, cols);
        murty::SparseMatrix<Scalar, Idx> smat(mat, max_per_row, max_val);
        murty::Assignment<Scalar, Idx> ass;

        {
            py::gil_scoped_release release;
            if (W) ass = murty::assignment<Scalar, Idx, murty::SparseMatrix<Scalar, Idx>>(smat, *W, max_cost);
            else ass = murty::assignment<Scalar, Idx, murty::SparseMatrix<Scalar, Idx>>(smat, max_cost);
        }

        return ass;

    }, "Solves sparse assignment problem", 
       py::arg("C"), 
       py::arg("max_per_row") = std::numeric_limits<size_t>::max(),
       py::arg("max_val") = std::numeric_limits<Scalar>::max(),
       py::arg("workers") = py::none(),
       py::arg("max_cost") = std::numeric_limits<Scalar>::max());
}

template <typename Scalar, typename Idx>
void bind_murty(py::module& m, const std::string& name) {
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    m.def(name.c_str(), [](const ArrT& C, size_t K, const std::vector<std::vector<Idx>>& row_subsets,
                           const std::vector<std::vector<Idx>>& col_subsets, const std::vector<Scalar>& base_costs, murty::MurtyWorkers<Scalar, Idx>* W, Scalar max_cost) {
        auto info = C.request();
        if (info.ndim != 2)
            throw std::invalid_argument("C must be a 2D matrix");

        size_t rows = info.shape[0];
        size_t cols = info.shape[1];
        
        murty::DenseMatrixView<Scalar> mat(static_cast<const Scalar*>(info.ptr), rows, cols);
        std::vector<murty::Assignment<Scalar, Idx>> results;

        {
            py::gil_scoped_release release;
            if (W) results = murty::solve_subsets<Scalar, Idx, murty::DenseMatrixView<Scalar>>(mat, K, *W, row_subsets, col_subsets, base_costs, max_cost);
            else results = murty::solve_subsets<Scalar, Idx, murty::DenseMatrixView<Scalar>>(mat, K, row_subsets, col_subsets, base_costs, max_cost);
        }

        return results;

    }, "Solves K-best assignments using Murty's algorithm", 
       py::arg("C"), py::arg("K"), 
       py::arg("row_subsets") = std::vector<std::vector<Idx>>{},
       py::arg("col_subsets") = std::vector<std::vector<Idx>>{},
       py::arg("base_costs") = std::vector<Scalar>{},
       py::arg("workers") = py::none(),
       py::arg("max_cost") = std::numeric_limits<Scalar>::max());
}

template <typename Scalar, typename Idx>
void bind_murty_smat(py::module& m, const std::string& name) {
    using T = murty::SparseMatrix<Scalar, Idx>;

    m.def(name.c_str(), [](const T& smat, size_t K, const std::vector<std::vector<Idx>>& row_subsets,
                           const std::vector<std::vector<Idx>>& col_subsets, const std::vector<Scalar>& base_costs, murty::MurtyWorkers<Scalar, Idx>* W, Scalar max_cost) {

        std::vector<murty::Assignment<Scalar, Idx>> results;

        {
            py::gil_scoped_release release;
            if (W) results = murty::solve_subsets<Scalar, Idx, T>(smat, K, *W, row_subsets, col_subsets, base_costs, max_cost);
            else results = murty::solve_subsets<Scalar, Idx, T>(smat, K, row_subsets, col_subsets, base_costs, max_cost);
        }

        return results;

    }, "Solves K-best assignments using Murty's algorithm", 
       py::arg("C"), py::arg("K"), 
       py::arg("row_subsets") = std::vector<std::vector<Idx>>{},
       py::arg("col_subsets") = std::vector<std::vector<Idx>>{},
       py::arg("base_costs") = std::vector<Scalar>{},
       py::arg("workers") = py::none(),
       py::arg("max_cost") = std::numeric_limits<Scalar>::max());
}

template <typename Scalar, typename Idx>
void bind_murty_sparse(py::module& m, const std::string& name) {
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    m.def(name.c_str(), [](const ArrT& C, size_t K, size_t max_per_row, Scalar max_val,
                           const std::vector<std::vector<Idx>>& row_subsets, const std::vector<std::vector<Idx>>& col_subsets,
                           const std::vector<Scalar>& base_costs, murty::MurtyWorkers<Scalar, Idx>* W, Scalar max_cost) {

        auto info = C.request();
        if (info.ndim != 2)
            throw std::invalid_argument("C must be a 2D matrix");
            
        size_t rows = info.shape[0];
        size_t cols = info.shape[1];
        
        murty::DenseMatrixView<Scalar> mat(static_cast<const Scalar*>(info.ptr), rows, cols);
        murty::SparseMatrix<Scalar, Idx> smat(mat, max_per_row, max_val);
        std::vector<murty::Assignment<Scalar, Idx>> results;

        {
            py::gil_scoped_release release;
            if (W) results = murty::solve_subsets<Scalar, Idx, murty::SparseMatrix<Scalar, Idx>>(smat, K, *W, row_subsets, col_subsets, base_costs, max_cost);
            else results = murty::solve_subsets<Scalar, Idx, murty::SparseMatrix<Scalar, Idx>>(smat, K, row_subsets, col_subsets, base_costs, max_cost);
        }
        
        return results;
        
    }, "Solves K-best assignments using Murty's algorithm", 
       py::arg("C"), py::arg("K"),
       py::arg("max_per_row") = std::numeric_limits<size_t>::max(),
       py::arg("max_val") = std::numeric_limits<Scalar>::max(),
       py::arg("row_subsets") = std::vector<std::vector<Idx>>{},
       py::arg("col_subsets") = std::vector<std::vector<Idx>>{},
       py::arg("base_costs") = std::vector<Scalar>{},
       py::arg("workers") = py::none(),
       py::arg("max_cost") = std::numeric_limits<Scalar>::max());
}

template <typename Scalar, typename Idx>
void bind_murty_fns(py::module& m) {
    bind_assignment<Scalar, Idx>(m, "assignment");
    bind_assignment_sparse<Scalar, Idx>(m, "assignment_sparse");
    bind_murty<Scalar, Idx>(m, "murty");
    bind_murty_sparse<Scalar, Idx>(m, "murty_sparse");
    bind_murty_smat<Scalar, Idx>(m, "murty");
}

using MurtyIdx = int;

template <typename Scalar, typename Idx>
void bind_murty_structs_all(py::module& m, const std::string& suffix) {
    bind_assignment_workers<Scalar, Idx>(m, "_AssignmentWorkers" + suffix);
    bind_murty_workers<Scalar, Idx>(m, "_MurtyWorkers" + suffix);
    bind_assignment_struct<Scalar, Idx>(m, "_Assignment" + suffix);
    bind_sparse_matrix<Scalar, Idx>(m, "_SparseMatrix" + suffix);
}

template <typename T>
void bind_murty(py::module& m, const std::string& type_str) {

    std::string type_suf = "_" + type_str;

    bind_murty_fns<T, MurtyIdx>(m);
    bind_murty_structs_all<T, MurtyIdx>(m, type_suf);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Bindings for Murty's algortihm";

    bind_murty<double>(m, "d");
    bind_murty<float>(m, "f");

    m.attr("ASSIGNMENT_UNMATCHED") = murty::Subproblem<double, MurtyIdx>::UNMATCHED;
    m.attr("ASSIGNMENT_EMPTY") = murty::Subproblem<double, MurtyIdx>::EMPTY;
}
