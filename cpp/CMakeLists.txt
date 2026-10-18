# planetmodel's C++ library: header-only readers and writers of what the
# Python writes, and the interpolation of its rules.
#
# A code fetches it at a release tag and links one target:
#
#   FetchContent_Declare(planetmodel
#     GIT_REPOSITORY https://github.com/da380/planetmodel.git
#     GIT_TAG        v<version>
#     SOURCE_SUBDIR  cpp)
#   FetchContent_MakeAvailable(planetmodel)
#   target_link_libraries(my_code PRIVATE planetmodel::planetmodel)
#
# The model file needs netCDF-C, behind PLANETMODEL_WITH_NETCDF: on by
# default and then required, so that a build lacking it fails at
# configuration and says so; off, nothing of netCDF is looked for and the
# headers that need it refuse to compile.
cmake_minimum_required(VERSION 3.24)

# The version is the Python package's, read from the one place it is set.
set(_pm_pyproject "${CMAKE_CURRENT_LIST_DIR}/../pyproject.toml")
if(NOT EXISTS "${_pm_pyproject}")
  message(FATAL_ERROR
    "planetmodel: ${_pm_pyproject} is missing; the C++ library is fetched "
    "with the whole repository and SOURCE_SUBDIR cpp")
endif()
file(STRINGS "${_pm_pyproject}" _pm_version_line
     REGEX "^version = \"[^\"]+\"$" LIMIT_COUNT 1)
string(REGEX REPLACE "^version = \"([^\"]+)\"$" "\\1" PLANETMODEL_VERSION_STRING
       "${_pm_version_line}")
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" _pm_version_numbers
       "${PLANETMODEL_VERSION_STRING}")
if(NOT _pm_version_numbers)
  message(FATAL_ERROR
    "planetmodel: no version found in ${_pm_pyproject}")
endif()

project(planetmodel VERSION ${_pm_version_numbers} LANGUAGES CXX)

option(PLANETMODEL_WITH_NETCDF
       "Read and write model files, over netCDF-C (required when on)" ON)
option(PLANETMODEL_BUILD_TESTS
       "Build planetmodel's own tests" ${PROJECT_IS_TOP_LEVEL})
option(PLANETMODEL_WARNINGS_AS_ERRORS
       "Treat warnings in planetmodel's own tests as errors" OFF)
set(PLANETMODEL_INTERPOLATION_TAG v1.1.1 CACHE STRING
    "The tag of da380/Interpolation to fetch")

include(FetchContent)

# Interpolation brings NumericConcepts with it, at the tag it pins; a code
# that declared either first keeps its own declaration.
FetchContent_Declare(Interpolation
  GIT_REPOSITORY https://github.com/da380/Interpolation.git
  GIT_TAG        ${PLANETMODEL_INTERPOLATION_TAG}
  FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(Interpolation)

add_library(planetmodel INTERFACE)
add_library(planetmodel::planetmodel ALIAS planetmodel)
target_include_directories(planetmodel INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(planetmodel INTERFACE cxx_std_23)
target_link_libraries(planetmodel INTERFACE Interpolation::Interpolation)
target_compile_definitions(planetmodel INTERFACE
  "PLANETMODEL_VERSION=\"${PLANETMODEL_VERSION_STRING}\"")

if(PLANETMODEL_WITH_NETCDF)
  # netCDF-C's own CMake package first, then pkg-config; the imported
  # target is made global so that the codes linking planetmodel, in other
  # directories, see it too.
  set(_pm_netcdf "")
  find_package(netCDF CONFIG QUIET)
  if(TARGET netCDF::netcdf)
    set(_pm_netcdf netCDF::netcdf)
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(PLANETMODEL_NETCDF QUIET IMPORTED_TARGET GLOBAL netcdf)
      if(TARGET PkgConfig::PLANETMODEL_NETCDF)
        set(_pm_netcdf PkgConfig::PLANETMODEL_NETCDF)
      endif()
    endif()
  endif()
  if(NOT _pm_netcdf)
    message(FATAL_ERROR
      "planetmodel: netCDF-C was not found, neither by find_package(netCDF "
      "CONFIG) (netCDF_DIR or CMAKE_PREFIX_PATH) nor by pkg-config (netcdf.pc "
      "on PKG_CONFIG_PATH). Install it and point CMake at it, or set "
      "PLANETMODEL_WITH_NETCDF=OFF to build without the model file.")
  endif()
  get_target_property(_pm_netcdf_imported ${_pm_netcdf} IMPORTED)
  if(_pm_netcdf_imported)
    set_target_properties(${_pm_netcdf} PROPERTIES IMPORTED_GLOBAL TRUE)
  endif()
  target_link_libraries(planetmodel INTERFACE ${_pm_netcdf})
  target_compile_definitions(planetmodel INTERFACE PLANETMODEL_WITH_NETCDF=1)
  message(STATUS "planetmodel: the model file over ${_pm_netcdf}")
else()
  message(STATUS "planetmodel: built without netCDF-C, so without the model file")
endif()

if(PLANETMODEL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
