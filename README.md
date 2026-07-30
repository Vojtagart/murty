# Murty

A high-performance implementation of Murty's algorithm for finding the K-best solutions to the linear assignment problem. Note that the reported solutions may left some of the rows or columns unassigned with the cost of zero. To prevent that, simply subtract $\max(c_{ij}) + 1$ from all the elements of the cost matrix $c$

It provides a header-only C++20 backend and a Python frontend bound via `pybind11` and managed by `scikit-build-core`.

## Project Structure

* `include/murty/`: Core C++ header-only library containing the SSP solver, Murty's algorithm, and custom matrix structures.
* `bindings/`: Python frontend and backend interface utilizing `pybind11`.
* `benchmark/`: C++ benchmarking suite comparing accuracy and execution time against `motrom/fastmurty` for both dense and sparse matrices.
* `tests/`: Unit tests written in Python using `pytest` to verify the bindings and algorithmic correctness.
* `scripts/`: Shell scripts for automating the C++ build process and executing benchmarks.

## Requirements

### System Dependencies
* CMake >= 3.14
* A C++20 compatible compiler

### Python Dependencies

**Core Library:**
* Python >= 3.8
* `numpy` >= 1.20.0

**Build System:**
* `pybind11`
* `scikit-build-core`

**Testing:**
* `pytest`

## Installation

### Python (via pip)
The Python library is built and installed using standard packaging tools. `scikit-build-core` automatically invokes CMake to compile the C++ backend.

```bash
# Install in the current environment
pip install .

# Or install in editable mode for development
pip install -e .
```

By default, `pyproject.toml` is configured to build in Release mode, which applies standard `-O3` (or `/O2` for MSVC) optimizations for optimal performance.

### C++ (via CMake FetchContent)
Because the core library is header-only, it does not require compilation. You can integrate it directly into an existing CMake project:

```cmake
include(FetchContent)
FetchContent_Declare(
    murty
    GIT_REPOSITORY [https://github.com/Vojtagart/murty.git](https://github.com/Vojtagart/murty.git)
    GIT_TAG        main
)
FetchContent_MakeAvailable(murty)

target_link_libraries(your_target PRIVATE murty)
```

## Usage

### Python Usage

Once installed, you can use `murty` directly with NumPy arrays to solve single-assignment or K-best assignment problems.

```python
import numpy as np
import murty

C = np.array([
    [4.0, 1.0, 3.0],
    [2.0, 0.5, 5.0],
    [3.0, 3.0, 2.0]
], dtype=np.float64)

# Find the single optimal assignment
best = murty.assignment(C)
print("Optimal Cost:", best.cost)
print("Assignment pairs [row, col]:\n", best.ass)

# Find the K-best assignments
results = murty.murty(C, K=3)
for i, assignment in enumerate(results):
    print(f"Rank {i+1} Cost: {assignment.cost}")
```

### C++ Usage

Include the core solving header in your C++ code to leverage the header-only engine directly.

```cpp
#include <iostream>
#include <vector>
#include <span>
#include "murty/solve.hpp"

int main() {
    double data[9] = {
        4.0, 1.0, 3.0,
        2.0, 0.5, 5.0,
        3.0, 3.0, 2.0
    };
    murty::DenseMatrixView<double> C(data, 3, 3);

    // Find the single optimal assignment
    auto best = murty::assignment(C);
    std::cout << "Optimal Cost: " << best.cost << "\n";

    // Find the K-best assignments
    std::span<const decltype(C)> C_span(&C, 1);
    auto [assignments, idxs] = murty::solve(C_span, 3);

    for (size_t i = 0; i < assignments.size(); ++i) {
        std::cout << "Rank " << (i + 1) << " Cost: " << assignments[i].cost << "\n";
    }
}
```

## Testing and Benchmarking

### Python Tests

The test suite validates the Python bindings and edge cases. Ensure the package is installed in editable mode first.

```bash
pip install -e .
pytest tests/python
```

### C++ Benchmarks

The benchmarking suite automatically downloads fastmurty, compiles it into a static library, and executes performance comparisons for both dense and sparse matrices.

```bash
# Configure and build the benchmarks in Release mode
./scripts/build.sh build Release

# Execute the benchmark suite
./scripts/run_benchmarks.sh build
```
