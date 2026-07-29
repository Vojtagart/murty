cmake_minimum_required(VERSION 3.14)
project(Murty LANGUAGES CXX C)

add_library(murty INTERFACE)
target_include_directories(murty INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(murty INTERFACE cxx_std_20)

# Options
option(MURTY_BUILD_TESTS "Build correctness tests" OFF)
option(MURTY_BUILD_BENCHMARKS "Build benchmarks against other versions" OFF)
option(MURTY_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(MURTY_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()

if(MURTY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MURTY_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()