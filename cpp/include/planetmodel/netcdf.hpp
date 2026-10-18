#ifndef PLANETMODEL_NETCDF_HPP
#define PLANETMODEL_NETCDF_HPP

/**
 * @file
 * @brief netCDF-C, over which the model file is read and written.
 *
 * It is there when the library was configured with
 * PLANETMODEL_WITH_NETCDF, the default; a code built without it cannot
 * include this header, nor any that includes it.  `netcdf::File` opens a
 * file for reading and closes it when it goes; a group of it, the file
 * itself included, is an `int` id as netCDF-C has it.  The readers take a
 * group, a name and the source the errors name, and refuse with an
 * `Error` naming the variable or attribute.
 */

#ifndef PLANETMODEL_WITH_NETCDF
#error "planetmodel was configured without netCDF-C, so without the model file: configure it with PLANETMODEL_WITH_NETCDF=ON"
#endif

#include <netcdf.h>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel::netcdf {

/**
 * @brief A netCDF-C status turned into an `Error` where it is a fault.
 * @param status What a netCDF-C call returned.
 * @param source The file, as the error names it.
 * @param where The variable or attribute, as the error names it.
 * @throws planetmodel::Error with netCDF-C's message where `status` is a
 *         fault.
 */
inline void Check(int status, const std::string& source, std::string_view where) {
  if (status != NC_NOERR) throw Error(source, where, nc_strerror(status));
}

/** @brief A file open for reading, closed when it goes. */
class File {
 public:
  /**
   * @brief Open a file for reading.
   * @param path The file.
   * @throws planetmodel::Error if netCDF-C cannot open it.
   */
  explicit File(const std::filesystem::path& path) : path_(path.string()) {
    Check(nc_open(path_.c_str(), NC_NOWRITE, &id_), path_, "");
  }
  /** @brief A file is not copied, its handle being its own. */
  File(const File&) = delete;
  /** @brief A file is not copied, its handle being its own. */
  File& operator=(const File&) = delete;
  /** @brief The handle moves with the file. */
  File(File&& other) noexcept : path_(std::move(other.path_)), id_(std::exchange(other.id_, -1)) {}
  /** @brief The handles are exchanged. */
  File& operator=(File&& other) noexcept {
    std::swap(path_, other.path_);
    std::swap(id_, other.id_);
    return *this;
  }
  /** @brief Close the file. */
  ~File() {
    if (id_ >= 0) nc_close(id_);
  }

  /** @brief The file's own id, the root group. */
  [[nodiscard]] int Id() const noexcept { return id_; }
  /** @brief The path it was opened from, as errors name it. */
  [[nodiscard]] const std::string& Path() const noexcept { return path_; }

 private:
  std::string path_;
  int id_ = -1;
};

/** @brief The values of a variable, flattened in C order, and its shape. */
struct Array {
  std::vector<double> values{};    /**< The values, last index fastest. */
  std::vector<std::size_t> shape{}; /**< The length of each dimension. */
};

/**
 * @brief Whether a group holds a variable.
 * @param group The group, or the file's id.
 * @param name The variable.
 */
[[nodiscard]] inline bool HasVariable(int group, std::string_view name) {
  auto id = 0;
  return nc_inq_varid(group, std::string{name}.c_str(), &id) == NC_NOERR;
}

/**
 * @brief The id of a subgroup.
 * @param group The group, or the file's id.
 * @param name The subgroup.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error if the group has no such subgroup.
 */
[[nodiscard]] inline int Group(int group, std::string_view name, const std::string& source) {
  auto id = 0;
  Check(nc_inq_grp_ncid(group, std::string{name}.c_str(), &id), source, name);
  return id;
}

/**
 * @brief The id of a variable.
 * @param group The group, or the file's id.
 * @param name The variable.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error naming the variable where the group has none.
 */
[[nodiscard]] inline int Variable(int group, std::string_view name, const std::string& source) {
  auto id = 0;
  if (nc_inq_varid(group, std::string{name}.c_str(), &id) != NC_NOERR) {
    throw Error(source, name, "is not a variable of the file");
  }
  return id;
}

/**
 * @brief A variable's values as doubles, and its shape.
 * @param group The group, or the file's id.
 * @param name The variable.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error if there is no such variable or it cannot be
 *         read as doubles.
 */
[[nodiscard]] inline Array ReadDoubles(int group, std::string_view name,
                                       const std::string& source) {
  const auto var = Variable(group, name, source);
  auto ndims = 0;
  Check(nc_inq_varndims(group, var, &ndims), source, name);
  auto dims = std::vector<int>(static_cast<std::size_t>(ndims));
  Check(nc_inq_vardimid(group, var, dims.data()), source, name);
  auto out = Array{};
  auto size = std::size_t{1};
  for (const auto dim : dims) {
    auto length = std::size_t{0};
    Check(nc_inq_dimlen(group, dim, &length), source, name);
    out.shape.push_back(length);
    size *= length;
  }
  out.values.resize(size);
  if (size > 0) Check(nc_get_var_double(group, var, out.values.data()), source, name);
  return out;
}

/** @brief The variable id of a group's own attributes. */
inline constexpr auto GLOBAL = NC_GLOBAL;

/**
 * @brief Whether an attribute exists.
 * @param group The group, or the file's id.
 * @param var The variable's id, or `GLOBAL`.
 * @param name The attribute.
 */
[[nodiscard]] inline bool HasAttribute(int group, int var, std::string_view name) {
  auto type = nc_type{};
  auto length = std::size_t{0};
  return nc_inq_att(group, var, std::string{name}.c_str(), &type, &length) == NC_NOERR;
}

/**
 * @brief An attribute as text, from a character or a string attribute.
 * @param group The group, or the file's id.
 * @param var The variable's id, or `GLOBAL`.
 * @param name The attribute.
 * @param source The file, as an error names it.
 * @return The text, or empty where there is no such attribute.
 * @throws planetmodel::Error if the attribute is not text.
 */
[[nodiscard]] inline std::optional<std::string> TextAttribute(int group, int var,
                                                              std::string_view name,
                                                              const std::string& source) {
  auto type = nc_type{};
  auto length = std::size_t{0};
  const auto key = std::string{name};
  if (nc_inq_att(group, var, key.c_str(), &type, &length) != NC_NOERR) return std::nullopt;
  if (type == NC_CHAR) {
    auto text = std::string(length, '\0');
    Check(nc_get_att_text(group, var, key.c_str(), text.data()), source, name);
    return text;
  }
  if (type == NC_STRING && length == 1) {
    char* text = nullptr;
    Check(nc_get_att_string(group, var, key.c_str(), &text), source, name);
    auto out = std::string{text ? text : ""};
    nc_free_string(1, &text);
    return out;
  }
  throw Error(source, name, "is not a text attribute");
}

/**
 * @brief An attribute as numbers.
 * @param group The group, or the file's id.
 * @param var The variable's id, or `GLOBAL`.
 * @param name The attribute.
 * @param source The file, as an error names it.
 * @return The numbers, or empty where there is no such attribute.
 * @throws planetmodel::Error if the attribute is text.
 */
[[nodiscard]] inline std::optional<std::vector<double>> NumberAttribute(
    int group, int var, std::string_view name, const std::string& source) {
  auto type = nc_type{};
  auto length = std::size_t{0};
  const auto key = std::string{name};
  if (nc_inq_att(group, var, key.c_str(), &type, &length) != NC_NOERR) return std::nullopt;
  if (type == NC_CHAR || type == NC_STRING) {
    throw Error(source, name, "is text, not a number");
  }
  auto out = std::vector<double>(length);
  Check(nc_get_att_double(group, var, key.c_str(), out.data()), source, name);
  return out;
}

}  // namespace planetmodel::netcdf

#endif  // PLANETMODEL_NETCDF_HPP
