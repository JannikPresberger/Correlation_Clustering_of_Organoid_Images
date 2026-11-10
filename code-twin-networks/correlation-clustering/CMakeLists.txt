cmake_minimum_required(VERSION 3.12...3.31)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

project(correlation-clustering-of-organoid-images CXX)

# The bundled andres headers rely on std::allocator members removed in C++20,
# and compilers such as GCC >= 16 default to C++20.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

include_directories(include)

# Gurobi
find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})

find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
link_directories( ${HDF5_LIBRARY_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )

add_executable(correlation-clustering-subsets src/correlation-clustering-subsets.cxx)
target_include_directories(correlation-clustering-subsets PUBLIC include)
target_link_libraries(correlation-clustering-subsets ${HDF5_CXX_LIBRARIES} )
target_link_libraries(correlation-clustering-subsets ${GUROBI_LIBRARIES} )