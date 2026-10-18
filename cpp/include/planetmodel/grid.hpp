#ifndef PLANETMODEL_GRID_HPP
#define PLANETMODEL_GRID_HPP

/**
 * @file
 * @brief Angular grids, as `planetmodel.sampling` defines them.
 *
 * A grid is a product of colatitudes, strictly increasing in [0, pi], and
 * longitudes, strictly increasing in [0, 2 pi), both in radians.  Either
 * pole may be a node: a colatitude of exactly 0 or pi.  `kind` names the
 * family a grid belongs to and `lmax` the band it resolves, where it has
 * one.
 */

#include <array>
#include <cmath>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel {

/** @brief The families of angular grid the files name. */
inline constexpr auto GRID_KINDS = std::array<std::string_view, 4>{
    "gauss_legendre", "driscoll_healy", "equiangular", "custom"};

/** @brief A product angular node set: colatitudes times longitudes. */
class AngularGrid {
 public:
  /**
   * @brief A grid of the given nodes.
   * @param colatitudes Strictly increasing, in [0, pi].
   * @param longitudes Strictly increasing, in [0, 2 pi).
   * @param kind One of `GRID_KINDS`.
   * @param lmax The band the grid resolves, where it has one.
   * @throws planetmodel::Error if the nodes, the kind or the band are not
   *         as the module says.
   */
  AngularGrid(std::vector<double> colatitudes, std::vector<double> longitudes,
              std::string kind = "custom", std::optional<int> lmax = std::nullopt)
      : colatitudes_(std::move(colatitudes)),
        longitudes_(std::move(longitudes)),
        kind_(std::move(kind)),
        lmax_(lmax) {
    constexpr auto pi = std::numbers::pi;
    if (colatitudes_.empty() || longitudes_.empty()) Fail("needs nodes in both directions");
    for (const auto* nodes : {&colatitudes_, &longitudes_}) {
      for (auto i = std::size_t{0}; i < nodes->size(); ++i) {
        if (!std::isfinite((*nodes)[i])) Fail("the nodes must be finite");
        if (i > 0 && !((*nodes)[i] > (*nodes)[i - 1])) {
          Fail(nodes == &colatitudes_ ? "colatitudes must be strictly increasing"
                                      : "longitudes must be strictly increasing");
        }
      }
    }
    if (colatitudes_.front() < 0.0 || colatitudes_.back() > pi) {
      Fail("colatitudes must lie in [0, pi]");
    }
    if (longitudes_.front() < 0.0 || longitudes_.back() >= 2.0 * pi) {
      Fail("longitudes must lie in [0, 2 pi)");
    }
    auto known = false;
    for (const auto k : GRID_KINDS) known = known || k == kind_;
    if (!known) Fail(std::format("{} is not a kind of grid", kind_));
    if (lmax_ && *lmax_ < 0) Fail(std::format("lmax must be non-negative, got {}", *lmax_));
  }

  /** @brief The colatitudes, in radians. */
  [[nodiscard]] const std::vector<double>& Colatitudes() const noexcept { return colatitudes_; }
  /** @brief The longitudes, in radians. */
  [[nodiscard]] const std::vector<double>& Longitudes() const noexcept { return longitudes_; }
  /** @brief The number of colatitudes. */
  [[nodiscard]] std::size_t NTheta() const noexcept { return colatitudes_.size(); }
  /** @brief The number of longitudes. */
  [[nodiscard]] std::size_t NPhi() const noexcept { return longitudes_.size(); }
  /** @brief The family the grid belongs to, one of `GRID_KINDS`. */
  [[nodiscard]] const std::string& Kind() const noexcept { return kind_; }
  /** @brief The band the grid resolves, where it has one. */
  [[nodiscard]] const std::optional<int>& Lmax() const noexcept { return lmax_; }

  /** @brief Whether the north pole is a node. */
  [[nodiscard]] bool NorthPoleNode() const noexcept { return colatitudes_.front() == 0.0; }
  /** @brief Whether the south pole is a node. */
  [[nodiscard]] bool SouthPoleNode() const noexcept {
    return colatitudes_.back() == std::numbers::pi;
  }

 private:
  [[noreturn]] static void Fail(std::string_view message) {
    throw Error("AngularGrid", "", message);
  }

  std::vector<double> colatitudes_;
  std::vector<double> longitudes_;
  std::string kind_;
  std::optional<int> lmax_;
};

}  // namespace planetmodel

#endif  // PLANETMODEL_GRID_HPP
