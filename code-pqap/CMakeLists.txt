cmake_minimum_required(VERSION 3.0.0)
project(Organoid_PQA VERSION 0.1.0)

#

if(CXX)
  enable_language(CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MARCH_FLAG "" CACHE STRING "compiler flag to pass e.g. -march=haswell")

set(CXX_FLAGS_RELEASE_TMP  ${CMAKE_CXX_FLAGS_RELEASE})

string(APPEND CXX_FLAGS_RELEASE_TMP " ")
string(APPEND CXX_FLAGS_RELEASE_TMP ${CMAKE_MARCH_FLAG})

set(CMAKE_CXX_FLAGS_RELEASE ${CXX_FLAGS_RELEASE_TMP})

#message(STATUS "CXX_RELEASE_FLAGS: ${CXX_FLAGS_RELEASE_TMP}")


if(NOT UNIX)
	set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules")
endif()
	
if(UNIX)
	set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/unix")
endif()
	
find_package(HDF5)
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIR})
	message(STATUS "${HDF5_INCLUDE_DIR}")
endif()
	
	
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER mpicxx)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")


 

#include(CTest)
#enable_testing()

#find_package(Python COMPONENTS Interpreter Numpy Development REQUIRED)

find_package(Python REQUIRED Development Interpreter NumPy)

option(CXX "enable C++ compilation" ON)
find_package(GUROBI REQUIRED)


find_package(Threads REQUIRED)


find_package(Qt5 COMPONENTS Core)
find_package(Qt5 COMPONENTS Gui)
find_package(Qt5 COMPONENTS Widgets)
find_package(Qt5 COMPONENTS Test)
find_package(Qt5 COMPONENTS Concurrent)

message(STATUS "${Qt5_FOUND}")

find_package( OpenCV REQUIRED )
include_directories("${PROJECT_SOURCE_DIR}/include" ${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${GUROBI_INCLUDE_DIRS})

#set(CMAKE_CXX_COMPILER mpicxx)
#include_directories(SYSTEM ${MPI_INCLUDE_PATH})

message(STATUS "${OpenCV_INCLUDE_DIRS}")

set(SOURCE_FILES
	src/main.cpp 
	src/image_processing.cpp 
	src/utils.cpp 
	src/experiment_data_handler.cpp
	src/python_handler.cpp
	src/cost_definitions.cpp
	src/solvers.cpp
	src/memory_manager.cpp
	src/clustering.cpp
	src/task_handler.cpp
	src/mpi_handler.cpp
)

set(HEADER_FILES
	include/image_processing.h
	include/utils.h 
	include/experiment_data_handler.h
	include/python_handler.h
	include/global_data_types.h
	include/cost_definitions.h
	include/solvers.h
	include/memory_manager.h
	include/clustering.h
	include/task_handler.h
	include/mpi_handler.h
)

add_executable(Organoid_PQA ${SOURCE_FILES} ${HEADER_FILES})

message(STATUS "gurobi dir: ${GUROBI_DIR}")
message(STATUS "gurobi lib: ${GUROBI_LIBRARY}")
message(STATUS "gurobi cxx lib: ${GUROBI_CXX_LIBRARY}")
message(STATUS "gurobi cxx debug lib: ${GUROBI_CXX_DEBUG_LIBRARY}")

target_link_libraries(${PROJECT_NAME} PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME} PRIVATE Python::Python)
target_link_libraries(${PROJECT_NAME} PRIVATE Python::NumPy)
target_link_libraries(${PROJECT_NAME} PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_C_LIBRARIES})

#target_link_libraries(${PROJECT_NAME} PRIVATE  Qt5::Core)
#target_link_libraries(${PROJECT_NAME} PRIVATE  Qt5::Gui)
#target_link_libraries(${PROJECT_NAME} PRIVATE  Qt5::Widgets)
#target_link_libraries(${PROJECT_NAME} PRIVATE  Qt5::Test)
#target_link_libraries(${PROJECT_NAME} PRIVATE  Qt5::Concurrent)


if(CXX)
	#set(CMAKE_CXX_STANDARD 11)
	target_link_libraries(${PROJECT_NAME} PRIVATE optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${GUROBI_LIBRARY})

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")


#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
