cmake_minimum_required(VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

project(correlation-clustering-of-organoids)
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

add_executable(correlation-clustering-subsets-independent-decisions src/correlation-clustering-subsets-independent-decisions.cxx)
target_include_directories(correlation-clustering-subsets-independent-decisions PUBLIC include)
target_link_libraries(correlation-clustering-subsets-independent-decisions ${HDF5_CXX_LIBRARIES} )
target_link_libraries(correlation-clustering-subsets-independent-decisions ${GUROBI_LIBRARIES} )

# experiments
add_executable(correlation-clustering src/correlation-clustering.cxx)
target_include_directories(correlation-clustering PUBLIC include)
target_link_libraries(correlation-clustering ${HDF5_CXX_LIBRARIES} )

add_executable(correlation-clustering-classifier src/correlation-clustering-classifier.cxx)
target_include_directories(correlation-clustering-classifier PUBLIC include)
target_link_libraries(correlation-clustering-classifier ${HDF5_CXX_LIBRARIES} )


add_executable(accuracy-0-1 src/accuracy-0-1.cxx)
target_include_directories(accuracy-0-1 PUBLIC include)
target_link_libraries(accuracy-0-1 ${HDF5_CXX_LIBRARIES} )


