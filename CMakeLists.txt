cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example)

find_package(Torch REQUIRED)
find_package (Python COMPONENTS Interpreter Development)

add_subdirectory(pybind11)
pybind11_add_module(example_py example.cpp)
target_include_directories(example_py PRIVATE "${Python_INCLUDE_DIRS}")
target_link_libraries(example_py PRIVATE "${TORCH_LIBRARIES}")

add_executable(example example.cpp)
target_include_directories(example PRIVATE "${Python_INCLUDE_DIRS}")
target_link_libraries(example "${TORCH_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 14)
