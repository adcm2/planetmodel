#ifndef PLANETMODEL_TESTS_FIXTURES_HPP
#define PLANETMODEL_TESTS_FIXTURES_HPP

// The fixtures make_fixtures.py writes.  A test whose fixture is missing
// is skipped, naming the script, unless PLANETMODEL_REQUIRE_FIXTURES is
// set in the environment (as CI sets it), when it fails.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace planetmodel::testing {

inline std::filesystem::path FixturesDir() {
  return std::filesystem::path{PLANETMODEL_FIXTURES_DIR};
}

/// Where the writer's tests leave the files they write, made where it is
/// missing.
inline std::filesystem::path WrittenDir() {
  const auto out = std::filesystem::path{PLANETMODEL_WRITTEN_DIR};
  std::filesystem::create_directories(out);
  return out;
}

inline bool FixturesRequired() {
  return std::getenv("PLANETMODEL_REQUIRE_FIXTURES") != nullptr;
}

}  // namespace planetmodel::testing

/// The path of fixture `name`, or the test skipped (failed, when fixtures
/// are required) where the script has not written it.
#define PLANETMODEL_FIXTURE(path_variable, name)                                    \
  const auto path_variable = ::planetmodel::testing::FixturesDir() / (name);        \
  if (!std::filesystem::exists(path_variable)) {                                   \
    if (::planetmodel::testing::FixturesRequired()) {                               \
      FAIL() << "fixture " << path_variable << " is missing";                       \
    }                                                                               \
    GTEST_SKIP() << "fixture " << path_variable                                     \
                 << " is missing: run cpp/tests/make_fixtures.py";                 \
  }

#endif  // PLANETMODEL_TESTS_FIXTURES_HPP
