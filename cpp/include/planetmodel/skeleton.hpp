#ifndef PLANETMODEL_SKELETON_HPP
#define PLANETMODEL_SKELETON_HPP

/**
 * @file
 * @brief The skeleton of a layered body, as `planetmodel.skeleton`
 * defines it.
 *
 * A strictly increasing list of boundary radii, the innermost zero for a
 * full body and positive for a hollow one.  Layer i lies between
 * boundaries i and i + 1, counted from the centre.  Radii are the same to
 * a tolerance relative to the skeleton's span, `DEFAULT_RTOL` unless one
 * is given.
 */

#include <cmath>
#include <cstddef>
#include <format>
#include <string_view>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel {

/**
 * @brief The relative tolerance to which radii are the same by default: a
 * fraction of the skeleton's span.
 */
inline constexpr auto DEFAULT_RTOL = 1e-9;

/** @brief Boundary radii, centre outward. */
class Skeleton {
 public:
  /**
   * @brief A skeleton of its boundary radii.
   * @param boundaries At least two finite, non-negative, strictly
   *        increasing radii.
   * @throws planetmodel::Error if the radii are not so.
   */
  explicit Skeleton(std::vector<double> boundaries) : boundaries_(std::move(boundaries)) {
    if (boundaries_.size() < 2) Fail("needs at least two boundary radii");
    for (auto i = std::size_t{0}; i < boundaries_.size(); ++i) {
      const auto r = boundaries_[i];
      if (!std::isfinite(r)) Fail("boundary radii must be finite");
      if (r < 0.0) Fail("boundary radii must be non-negative");
      if (i > 0 && !(r > boundaries_[i - 1])) Fail("boundary radii must be strictly increasing");
    }
  }

  /** @brief The boundary radii, centre outward. */
  [[nodiscard]] const std::vector<double>& Boundaries() const noexcept { return boundaries_; }
  /** @brief The number of layers, one fewer than the boundaries. */
  [[nodiscard]] std::size_t NLayers() const noexcept { return boundaries_.size() - 1; }

  /**
   * @brief The radii of a layer's inner and outer boundaries.
   * @param i The layer, from the centre.
   * @throws planetmodel::Error if there is no such layer.
   */
  [[nodiscard]] std::pair<double, double> Interval(std::size_t i) const {
    if (i >= NLayers()) {
      throw Error("Skeleton", "", std::format("has {} layers, so no layer {}", NLayers(), i));
    }
    return {boundaries_[i], boundaries_[i + 1]};
  }

  /** @brief The outer radius less the inner: the length every tolerance refers to. */
  [[nodiscard]] double Span() const noexcept { return boundaries_.back() - boundaries_.front(); }

  /** @brief Whether the innermost boundary lies above the centre. */
  [[nodiscard]] bool IsHollow() const noexcept { return boundaries_.front() > 0.0; }

  /** @brief Skeletons are equal when their boundaries are. */
  friend bool operator==(const Skeleton&, const Skeleton&) = default;

 private:
  [[noreturn]] static void Fail(std::string_view message) {
    throw Error("Skeleton", "", message);
  }

  std::vector<double> boundaries_;
};

}  // namespace planetmodel

#endif  // PLANETMODEL_SKELETON_HPP
