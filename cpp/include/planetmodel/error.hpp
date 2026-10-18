#ifndef PLANETMODEL_ERROR_HPP
#define PLANETMODEL_ERROR_HPP

/// @file
/// The one error the library throws.
///
/// Reading, checking and construction throw `planetmodel::Error`, naming
/// the source (a file, or the object being built) and where in it the
/// fault lies; evaluation throws only where a query leaves the domain of
/// what is evaluated.  Interpolation's own `std::invalid_argument` passes
/// through unchanged where the library hands it data to build on.

#include <stdexcept>
#include <string>
#include <string_view>

namespace planetmodel {

/// A fault in what the library was given: the message, and the source and
/// the place in it that it concerns, either of which may be empty.
class Error : public std::runtime_error {
 public:
  /// A fault with no source or place to name.
  explicit Error(std::string_view message) : Error({}, {}, message) {}

  /// A fault in `source` (a file's path, or what is being built) at
  /// `where` (a key, a variable, a line), read as "source: where: message"
  /// with the empty parts left out.
  Error(std::string_view source, std::string_view where, std::string_view message)
      : std::runtime_error(Compose(source, where, message)),
        source_(source),
        where_(where) {}

  /// The file or object the fault is in; empty where none was named.
  [[nodiscard]] const std::string& Source() const noexcept { return source_; }

  /// The place within the source; empty where none was named.
  [[nodiscard]] const std::string& Where() const noexcept { return where_; }

 private:
  static std::string Compose(std::string_view source, std::string_view where,
                             std::string_view message) {
    auto out = std::string{};
    for (const auto part : {source, where}) {
      if (!part.empty()) {
        out.append(part);
        out.append(": ");
      }
    }
    out.append(message);
    return out;
  }

  std::string source_;
  std::string where_;
};

}  // namespace planetmodel

#endif  // PLANETMODEL_ERROR_HPP
