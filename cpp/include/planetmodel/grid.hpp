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
 * one; `weights`, where it has them, are colatitude weights against
 * sin(theta) dtheta, so that the integral over the sphere of f is about
 * sum_i w_i sum_j (2 pi / nphi) f_ij.  The three shipped families are made
 * by `GaussLegendre`, `DriscollHealy` and `Equiangular`, nodes, weights
 * and band as the Python makes them; any other grid is `custom`.
 */

#include <algorithm>
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
   * @param weights The colatitude weights, one per colatitude, where it has
   *        them.
   * @throws planetmodel::Error if the nodes, the kind, the band or the
   *         weights are not as the module says.
   */
  AngularGrid(std::vector<double> colatitudes, std::vector<double> longitudes,
              std::string kind = "custom", std::optional<int> lmax = std::nullopt,
              std::optional<std::vector<double>> weights = std::nullopt)
      : colatitudes_(std::move(colatitudes)),
        longitudes_(std::move(longitudes)),
        kind_(std::move(kind)),
        lmax_(lmax),
        weights_(std::move(weights)) {
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
    if (weights_) {
      if (weights_->size() != colatitudes_.size()) {
        Fail(std::format("weights must have one entry per colatitude, {}, got {}",
                         colatitudes_.size(), weights_->size()));
      }
      for (const auto w : *weights_) {
        if (!std::isfinite(w)) Fail("weights must be finite");
      }
    }
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
  /** @brief The colatitude weights, where the grid has them. */
  [[nodiscard]] const std::optional<std::vector<double>>& Weights() const noexcept {
    return weights_;
  }

  /** @brief Whether the north pole is a node. */
  [[nodiscard]] bool NorthPoleNode() const noexcept { return colatitudes_.front() == 0.0; }
  /** @brief Whether the south pole is a node. */
  [[nodiscard]] bool SouthPoleNode() const noexcept {
    return colatitudes_.back() == std::numbers::pi;
  }
  /** @brief Whether the grid lacks a pole, so that values there are given apart. */
  [[nodiscard]] bool LacksPole() const noexcept { return !NorthPoleNode() || !SouthPoleNode(); }

  /** @brief Grids are equal when their nodes, kind, band and weights are. */
  friend bool operator==(const AngularGrid&, const AngularGrid&) = default;

 private:
  [[noreturn]] static void Fail(std::string_view message) {
    throw Error("AngularGrid", "", message);
  }

  std::vector<double> colatitudes_;
  std::vector<double> longitudes_;
  std::string kind_;
  std::optional<int> lmax_;
  std::optional<std::vector<double>> weights_;
};

/**
 * @brief The Gauss-Legendre grid resolving degrees and orders up to `lmax`.
 *
 * `lmax + 1` Gauss-Legendre colatitudes in x = cos(theta), theta
 * increasing, with the Legendre weights, and `nphi` equally spaced
 * longitudes from zero, 2 lmax + 1 by default, the fewest that resolve the
 * orders.
 *
 * @param lmax The band, non-negative.
 * @param nphi The number of longitudes, at least 2 lmax + 1.
 * @throws planetmodel::Error if `lmax` is negative or `nphi` too few.
 */
[[nodiscard]] inline AngularGrid GaussLegendre(int lmax, std::optional<int> nphi = std::nullopt) {
  if (lmax < 0) {
    throw Error("GaussLegendre", "", std::format("lmax must be non-negative, got {}", lmax));
  }
  const auto need = 2 * lmax + 1;
  const auto columns = nphi.value_or(need);
  if (columns < need) {
    throw Error("GaussLegendre", "",
                std::format("nphi={} cannot resolve orders |m| <= {}: the trapezoid rule "
                            "separates them only for nphi >= 2 lmax + 1 = {}",
                            columns, lmax, need));
  }
  const auto n = static_cast<std::size_t>(lmax) + 1;
  // P_n and its slope at z, by the three-term recurrence
  auto legendre = [n](double z) {
    auto previous = 1.0;
    auto value = z;
    for (auto k = std::size_t{2}; k <= n; ++k) {
      const auto next = ((2.0 * static_cast<double>(k) - 1.0) * z * value -
                         (static_cast<double>(k) - 1.0) * previous) /
                        static_cast<double>(k);
      previous = value;
      value = next;
    }
    const auto slope = static_cast<double>(n) * (z * value - previous) / (z * z - 1.0);
    return std::pair{value, slope};
  };
  // the roots by Newton's method from Chebyshev's estimates, x decreasing
  // so that theta increases; the weights from the slope at each root
  auto x = std::vector<double>(n);
  auto w = std::vector<double>(n);
  for (auto i = std::size_t{0}; i < n; ++i) {
    auto z = std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) /
                      (static_cast<double>(n) + 0.5));
    for (auto iteration = 0; iteration < 100; ++iteration) {
      const auto [value, slope] = legendre(z);
      const auto step = value / slope;
      z -= step;
      if (std::abs(step) <= 1e-14) break;
    }
    const auto slope = legendre(z).second;
    x[i] = z;
    w[i] = 2.0 / ((1.0 - z * z) * slope * slope);
  }
  // symmetric about the equator, and summing to 2
  for (auto i = std::size_t{0}; i < n; ++i) {
    const auto j = n - 1 - i;
    if (i < j) {
      const auto xs = 0.5 * (x[i] - x[j]);
      const auto ws = 0.5 * (w[i] + w[j]);
      x[i] = xs;
      x[j] = -xs;
      w[i] = w[j] = ws;
    } else if (i == j) {
      x[i] = 0.0;
    }
  }
  auto sum = 0.0;
  for (const auto v : w) sum += v;
  for (auto& v : w) v *= 2.0 / sum;
  auto theta = std::vector<double>(n);
  for (auto i = std::size_t{0}; i < n; ++i) theta[i] = std::acos(x[i]);
  auto phi = std::vector<double>(static_cast<std::size_t>(columns));
  for (auto j = 0; j < columns; ++j) {
    phi[static_cast<std::size_t>(j)] = 2.0 * std::numbers::pi * j / columns;
  }
  return AngularGrid(std::move(theta), std::move(phi), "gauss_legendre", lmax, std::move(w));
}

/**
 * @brief The Driscoll-Healy grid resolving degrees and orders up to `lmax`.
 *
 * n = 2 lmax + 2 equally spaced colatitudes theta_i = pi i / n, from the
 * north pole, a node, to one step short of the south pole; n equally
 * spaced longitudes from zero, or 2 n with `sampling` 2.  The weights are
 * Driscoll and Healy's, exact for the polynomials in cos(theta) of degree
 * below n.
 *
 * @param lmax The band, non-negative.
 * @param sampling 1, or 2 for twice the longitudes.
 * @throws planetmodel::Error if `lmax` is negative or `sampling` neither.
 */
[[nodiscard]] inline AngularGrid DriscollHealy(int lmax, int sampling = 1) {
  if (lmax < 0) {
    throw Error("DriscollHealy", "", std::format("lmax must be non-negative, got {}", lmax));
  }
  if (sampling != 1 && sampling != 2) {
    throw Error("DriscollHealy", "", std::format("sampling must be 1 or 2, got {}", sampling));
  }
  const auto n = 2 * lmax + 2;
  auto theta = std::vector<double>(static_cast<std::size_t>(n));
  auto weights = std::vector<double>(static_cast<std::size_t>(n));
  for (auto k = 0; k < n; ++k) {
    const auto t = std::numbers::pi * k / n;
    auto sum = 0.0;
    for (auto j = 0; j < n / 2; ++j) sum += std::sin((2 * j + 1) * t) / (2 * j + 1);
    theta[static_cast<std::size_t>(k)] = t;
    weights[static_cast<std::size_t>(k)] = (4.0 / n) * std::sin(t) * sum;
  }
  const auto nphi = sampling * n;
  auto phi = std::vector<double>(static_cast<std::size_t>(nphi));
  for (auto j = 0; j < nphi; ++j) {
    phi[static_cast<std::size_t>(j)] = 2.0 * std::numbers::pi * j / nphi;
  }
  return AngularGrid(std::move(theta), std::move(phi), "driscoll_healy", lmax, std::move(weights));
}

/**
 * @brief Equally spaced colatitudes and longitudes, with weights and a
 * band.
 *
 * The colatitudes are the midpoints theta_i = pi (i + 1/2) / ntheta, with
 * the weights of Fejer's first rule, or with `poles` both poles and the
 * nodes between, theta_i = pi i / (ntheta - 1), with the Clenshaw-Curtis
 * weights; the longitudes are phi_j = 2 pi j / nphi.  The band is ntheta -
 * 1 for the midpoints and ntheta - 2 with the poles, and at most (nphi -
 * 1) / 2.
 *
 * @param ntheta The number of colatitudes, at least 2 with `poles`.
 * @param nphi The number of longitudes.
 * @param poles Whether both poles are nodes.
 * @throws planetmodel::Error if the numbers are too few.
 */
[[nodiscard]] inline AngularGrid Equiangular(int ntheta, int nphi, bool poles = false) {
  if (ntheta < 1 || nphi < 1) throw Error("Equiangular", "", "ntheta and nphi must be positive");
  if (poles && ntheta < 2) {
    throw Error("Equiangular", "",
                std::format("an equiangular grid with both poles needs ntheta >= 2, got {}",
                            ntheta));
  }
  constexpr auto pi = std::numbers::pi;
  auto theta = std::vector<double>(static_cast<std::size_t>(ntheta));
  auto weights = std::vector<double>(static_cast<std::size_t>(ntheta));
  auto band = 0;
  if (poles) {
    // the nodes as numpy's linspace makes them, the poles exactly; the
    // Clenshaw-Curtis weights
    const auto big_n = ntheta - 1;
    const auto step = pi / big_n;
    for (auto k = 0; k < ntheta; ++k) {
      theta[static_cast<std::size_t>(k)] = k == big_n ? pi : k * step;
      const auto t = pi * k / big_n;
      auto sum = 0.0;
      for (auto j = 1; j <= big_n / 2; ++j) {
        const auto b = 2 * j == big_n ? 1.0 : 2.0;
        sum += b * std::cos(2 * j * t) / (4 * j * j - 1);
      }
      const auto c = k == 0 || k == big_n ? 1.0 : 2.0;
      weights[static_cast<std::size_t>(k)] = (c / big_n) * (1.0 - sum);
    }
    band = ntheta - 2;
  } else {
    // the midpoints, with Fejer's first rule
    for (auto k = 0; k < ntheta; ++k) {
      const auto t = pi * (k + 0.5) / ntheta;
      auto sum = 0.0;
      for (auto j = 1; j <= ntheta / 2; ++j) sum += std::cos(2 * j * t) / (4 * j * j - 1);
      theta[static_cast<std::size_t>(k)] = t;
      weights[static_cast<std::size_t>(k)] = (2.0 / ntheta) * (1.0 - 2.0 * sum);
    }
    band = ntheta - 1;
  }
  auto phi = std::vector<double>(static_cast<std::size_t>(nphi));
  for (auto j = 0; j < nphi; ++j) phi[static_cast<std::size_t>(j)] = 2.0 * pi * j / nphi;
  return AngularGrid(std::move(theta), std::move(phi), "equiangular",
                     std::min(band, (nphi - 1) / 2), std::move(weights));
}

}  // namespace planetmodel

#endif  // PLANETMODEL_GRID_HPP
