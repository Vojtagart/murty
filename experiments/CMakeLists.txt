cmake_minimum_required(VERSION 3.14)

add_executable(experiment_sparsification experiment_sparsification.cpp)
target_compile_features(experiment_sparsification PRIVATE cxx_std_20)
target_link_libraries(experiment_sparsification PRIVATE murty)

