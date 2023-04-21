cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(gt-memory-check)

if(GTENSOR_DEVICE STREQUAL "cuda")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# add dependencies
include(cmake/CPM.cmake)
CPMFindPackage(NAME gtensor
               GITHUB_REPOSITORY wdmapp/gtensor
               GIT_TAG "main"
               OPTIONS "GTENSOR_ENABLE_BLAS ON"
                       "GTENSOR_ENABLE_SOLVER ON")

add_executable(axpy_memory_usage)
target_gtensor_sources(axpy_memory_usage PRIVATE axpy_memory_usage.cxx)
target_link_libraries(axpy_memory_usage gtensor::gtensor)
target_link_libraries(axpy_memory_usage gtensor::blas)

add_executable(solver_memory_usage)
target_gtensor_sources(solver_memory_usage PRIVATE solver_memory_usage.cxx)
target_link_libraries(solver_memory_usage gtensor::gtensor)
target_link_libraries(solver_memory_usage gtensor::gtsolver)

add_executable(lu_memory_usage)
target_gtensor_sources(lu_memory_usage PRIVATE lu_memory_usage.cxx)
target_link_libraries(lu_memory_usage gtensor::gtensor)
target_link_libraries(lu_memory_usage gtensor::blas)
