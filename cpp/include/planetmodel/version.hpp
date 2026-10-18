#ifndef PLANETMODEL_VERSION_HPP
#define PLANETMODEL_VERSION_HPP

/**
 * @file
 * @brief The library's version, which is the Python package's: the CMake
 * target reads it from `pyproject.toml` and defines it for every code it
 * links.
 */

#include <string_view>

#ifndef PLANETMODEL_VERSION
#error "planetmodel's version is defined by its CMake target: link planetmodel::planetmodel"
#endif

namespace planetmodel {

/**
 * @brief The version of the release the library was fetched at, as the
 * Python package states it (`1.3.0`, say).
 */
[[nodiscard]] constexpr std::string_view Version() noexcept {
  return PLANETMODEL_VERSION;
}

}  // namespace planetmodel

#endif  // PLANETMODEL_VERSION_HPP
