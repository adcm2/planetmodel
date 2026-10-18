# Two small programs: one writes a model file of fields of its own, the
# other reads any model file.  With the tests, the second reads what the
# first wrote.
foreach(_example IN ITEMS write_model read_model)
  add_executable(planetmodel_${_example} ${_example}.cpp)
  target_link_libraries(planetmodel_${_example} PRIVATE planetmodel::planetmodel)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(planetmodel_${_example} PRIVATE -Wall -Wextra -Wpedantic
      $<$<BOOL:${PLANETMODEL_WARNINGS_AS_ERRORS}>:-Werror>)
  endif()
endforeach()

if(PLANETMODEL_BUILD_TESTS)
  set(_pm_example_file ${CMAKE_CURRENT_BINARY_DIR}/example.nc)
  add_test(NAME example_write_model
           COMMAND planetmodel_write_model ${_pm_example_file})
  add_test(NAME example_read_model
           COMMAND planetmodel_read_model ${_pm_example_file})
  set_tests_properties(example_write_model PROPERTIES FIXTURES_SETUP example_file)
  set_tests_properties(example_read_model PROPERTIES FIXTURES_REQUIRED example_file)
endif()
