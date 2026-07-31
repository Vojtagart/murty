# Murty

A high-performance implementation of Murty's algorithm for finding the K-best solutions to the linear assignment problem.

Note that the reported solutions may leave some of the rows or columns unassigned with the cost of zero. To prevent that, simply subtract $\max(c_{ij}) + 1$ from all the elements of the cost matrix $c$. Shifting all elements to negative values makes assigning a pair strictly better than leaving it unassigned at the cost of $0$.

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
    GIT_REPOSITORY https://github.com/Vojtagart/murty.git
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

For additional examples, refer to `tests/python/`

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

# Sources

[1] MOTRO, Michael. Fastmurty: Murty’s Algorithm C Implementation [online]. 2019. [visited on 2026-03-05]. Available from: https://github.com/motrom/fastmurty/tree/master.

[2] MOTRO, Michael; GHOSH, Joydeep. Scaling Data Association for Hypothesis-Oriented MHT. In: 2019 22nd International Conference on Information Fusion (FUSION). Ottawa, ON, Canada: IEEE, 2019, pp. 1–8. Available from doi: 10.23919/FUSION43075.2019.9011203.

[3] MILLER, Michael L.; STONE, Harold S.; COX, Ingemar J. Optimizing Murty’s Ranked Assignment Method. IEEE Transactions on Aerospace and Electronic Systems. 1997, vol. 33, no. 3, pp. 851–862. Available from doi: 10.1109/7.599256.

[4] CROUSE, David F. On Implementing 2D Rectangular Assignment Algorithms. IEEE Transactions on Aerospace and Electronic Systems. 2016, vol. 52, no. 4, pp. 1679–1696. Available from doi: 10.1109/TAES.2016.140952.

[5] JONKER, Roy; VOLGENANT, Anton. A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems. Computing. 1987, vol. 38, no. 4, pp. 325–340. Available from doi: 10.1007/BF02278710.

[6] MURTY, Katta G. An Algorithm for Ranking All the Assignments in Order of Increasing Cost. Operations Research. 1968, vol. 16, no. 3, pp. 682687. Available from doi: 10.1287/opre.16.3.682

