find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core python/bindings.cpp)

target_link_libraries(_core PRIVATE murty)
target_compile_features(_core PRIVATE cxx_std_20)

install(TARGETS _core DESTINATION murty)