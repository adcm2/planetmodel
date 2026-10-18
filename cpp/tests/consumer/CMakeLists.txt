# A code that fetches planetmodel's C++ library as the documentation tells
# it to, here from a local checkout: `scales` uses what needs no netCDF,
# `netcdf` includes the netCDF header and calls netCDF-C.
cmake_minimum_required(VERSION 3.24)
project(planetmodel_consumer LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(planetmodel
  SOURCE_DIR    ${PLANETMODEL_ROOT}
  SOURCE_SUBDIR cpp)
FetchContent_MakeAvailable(planetmodel)

add_executable(scales scales.cpp)
target_link_libraries(scales PRIVATE planetmodel::planetmodel)

add_executable(netcdf netcdf.cpp)
target_link_libraries(netcdf PRIVATE planetmodel::planetmodel)
