cmake_minimum_required(VERSION 3.14)

include(FetchContent)

# ===============================================
# Fetching the fastmurty
# ===============================================

FetchContent_Declare(
    fastmurty
    GIT_REPOSITORY https://github.com/motrom/fastmurty.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(fastmurty)

# -----------------------------------------------
# Adding the dense version
# -----------------------------------------------

add_library(fastmurty_dense STATIC
    ${fastmurty_SOURCE_DIR}/da.c
    ${fastmurty_SOURCE_DIR}/queue.c
    ${fastmurty_SOURCE_DIR}/sspDense.c
    ${fastmurty_SOURCE_DIR}/murtysplitDense.c
    ${fastmurty_SOURCE_DIR}/subproblem.c
)
target_include_directories(fastmurty_dense PUBLIC ${fastmurty_SOURCE_DIR})

# -----------------------------------------------
# Adding the sparse version
# -----------------------------------------------

add_library(fastmurty_sparse STATIC
    ${fastmurty_SOURCE_DIR}/da.c
    ${fastmurty_SOURCE_DIR}/queue.c
    ${fastmurty_SOURCE_DIR}/sspSparse.c
    ${fastmurty_SOURCE_DIR}/murtysplitSparse.c
    ${fastmurty_SOURCE_DIR}/subproblem.c
)
target_include_directories(fastmurty_sparse PUBLIC ${fastmurty_SOURCE_DIR})
target_compile_definitions(fastmurty_sparse PUBLIC SPARSE)

# ===============================================
# Adding the dense and sparse benchmarks
# ===============================================

# -----------------------------------------------
# Dense
# -----------------------------------------------

set(TCL_SOURCES "3rd_party_code/TrackerComponentLibrary/ShortestPathCPP.cpp")

add_executable(benchmark_dense benchmark_dense.cpp ${TCL_SOURCES})
target_compile_features(benchmark_dense PRIVATE cxx_std_20)
target_link_libraries(benchmark_dense PRIVATE murty fastmurty_dense)

# -----------------------------------------------
# Sparse
# -----------------------------------------------

add_executable(benchmark_sparse benchmark_sparse.cpp)
target_compile_features(benchmark_sparse PRIVATE cxx_std_20)
target_link_libraries(benchmark_sparse PRIVATE murty fastmurty_sparse)

# -----------------------------------------------
# Dense vs Sparse
# -----------------------------------------------

add_executable(benchmark_dense_sparse benchmark_dense_sparse.cpp)
target_compile_features(benchmark_dense_sparse PRIVATE cxx_std_20)
target_link_libraries(benchmark_dense_sparse PRIVATE murty)
