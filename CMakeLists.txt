cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(gt-memory-check)

# add dependencies
include(cmake/CPM.cmake)
CPMFindPackage(NAME gtensor
               GITHUB_REPOSITORY wdmapp/gtensor
               GIT_TAG "main"
               OPTIONS "GTENSOR_ENABLE_BLAS ON")

add_executable(axpy_memory_usage)
target_sources(axpy_memory_usage PRIVATE axpy_memory_usage.cxx)
target_link_libraries(axpy_memory_usage gtensor::gtensor)
target_link_libraries(axpy_memory_usage gtensor::blas)
