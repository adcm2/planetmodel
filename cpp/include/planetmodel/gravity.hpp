#ifndef PLANETMODEL_GRAVITY_HPP
#define PLANETMODEL_GRAVITY_HPP

/**
 * @file
 * @brief Cached spherical mass and gravity for an in-memory radial model.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <numbers>
#include <numeric>
#include <vector>

#include "planetmodel/radial_model.hpp"

namespace planetmodel {

/** @brief Error controls for the adaptive mass integrals. */
struct GravityIntegrationSettings {
  double relative_tolerance = 1e-10; /**< Relative quadrature error bound. */
  double absolute_tolerance = 0.0; /**< Local absolute quadrature error criterion. */
  unsigned max_depth = 20; /**< Maximum bisections, capped at 30. */
  std::size_t max_evaluations = 200000; /**< Density evaluations per preparation or query. */
};

/** @brief Cached mass and gravity from radial density, in model units. */
class RadialGravity {
 public:
  /** @brief Prepare masses split at every model integration breakpoint. */
  explicit RadialGravity(const RadialModel& model,
                         GravityIntegrationSettings settings = {})
      : G_(model.G()), settings_(settings) {
    if (!(std::isfinite(settings_.relative_tolerance) && settings_.relative_tolerance > 0.0 &&
          std::isfinite(settings_.absolute_tolerance) && settings_.absolute_tolerance >= 0.0 &&
          settings_.max_depth > 0 && settings_.max_depth <= 30 && settings_.max_evaluations > 0)) {
      throw Error("RadialGravity", "settings", "tolerances and maximum depth are invalid");
    }
    auto work = Work{0, settings_.max_evaluations};
    layers_.reserve(model.Layers().size());
    for (std::size_t i = 0; i < model.Layers().size(); ++i) {
      const auto& source = model.Layer(i);
      if (!source.fields.contains("rho"))
        throw Error("RadialGravity", std::format("layer {}", i), "needs a radial rho field");
      LayerMass layer{source.fields.at("rho"), {}};
      const auto& points = model.IntegrationBreakpoints(i);
      for (std::size_t j = 1; j < points.size(); ++j) {
        BuildInterval(layer, points[j - 1], points[j], 0, work);
      }
      layers_.push_back(std::move(layer));
    }
  }

  /** @brief Mass enclosed by radius r, using layer i to select an interface side. */
  [[nodiscard]] double Mass(std::size_t i, double r) const {
    const auto& layer = layers_.at(i);
    CheckRadius(layer, i, r);
    if (layer.pieces.empty() || r == layer.pieces.front().lo) return
        layer.pieces.empty() ? 0.0 : layer.pieces.front().prefix;
    const auto it = std::upper_bound(layer.pieces.begin(), layer.pieces.end(), r,
        [](double x, const Piece& piece) { return x < piece.hi; });
    if (it == layer.pieces.end()) return layer.pieces.back().prefix + layer.pieces.back().mass;
    if (r <= it->lo) return it->prefix;
    auto work = Work{0, settings_.max_evaluations};
    const auto partial = Integrate(layer.rho, it->lo, r, settings_.max_depth, work);
    return it->prefix + partial;
  }

  /** @brief Enclosed mass at a model radius, zero throughout a hollow cavity. */
  [[nodiscard]] double Mass(double r) const {
    CheckModelRadius(r);
    if (r < layers_.front().rho.Interval().first) return 0.0;
    return Mass(LayerAt(r), r);
  }

  /** @brief Inward acceleration g=GM/r²; pass layer i for one-sided interfaces. */
  [[nodiscard]] double Gravity(std::size_t i, double r) const {
    CheckRadius(layers_.at(i), i, r);
    return r == 0.0 ? 0.0 : G_ * Mass(i, r) / (r * r);
  }

  /** @brief Inward acceleration at a model radius; hollow cavities have zero field. */
  [[nodiscard]] double Gravity(double r) const {
    CheckModelRadius(r);
    return r == 0.0 || r < layers_.front().rho.Interval().first
        ? 0.0 : G_ * Mass(r) / (r * r);
  }

  /** @brief Radial derivative of gravity, with the regular center limit. */
  [[nodiscard]] double GravityDerivative(std::size_t i, double r) const {
    const auto& layer = layers_.at(i);
    CheckRadius(layer, i, r);
    if (r == 0.0) return 4.0 * std::numbers::pi * G_ * Density(layer.rho, 0.0) / 3.0;
    return G_ * (4.0 * std::numbers::pi * Density(layer.rho, r) -
                 2.0 * Mass(i, r) / (r * r * r));
  }

  /** @brief Gravity derivative at a model radius, using the outward side of interfaces. */
  [[nodiscard]] double GravityDerivative(double r) const {
    CheckModelRadius(r);
    if (r < layers_.front().rho.Interval().first) return 0.0;
    if (r == 0.0) return GravityDerivative(0, r);
    return GravityDerivative(LayerAt(r), r);
  }

 private:
  struct Piece { double lo; double hi; double prefix; double mass; };
  struct LayerMass { RadialFunction rho; std::vector<Piece> pieces; };
  struct Estimate { double value; double error; };
  struct Work { std::size_t evaluations; std::size_t maximum; };

  static constexpr std::array<double, 8> x_ = {
      0.9914553711208126392, 0.9491079123427585245, 0.8648644233597690728,
      0.7415311855993944399, 0.5860872354676911303, 0.4058451513773971669,
      0.2077849550078984676, 0.0};
  static constexpr std::array<double, 8> wk_ = {
      0.02293532201052922496, 0.06309209262997855329, 0.1047900103222501838,
      0.1406532597155259187, 0.1690047266392679028, 0.1903505780647854099,
      0.2044329400752988924, 0.2094821410847278280};
  static constexpr std::array<double, 4> wg_ = {
      0.1294849661688696933, 0.2797053914892766679,
      0.3818300505051189449, 0.4179591836734693878};

  [[nodiscard]] double Integrand(const RadialFunction& rho, double r, Work& work) const {
    if (work.evaluations == work.maximum)
      throw Error("RadialGravity", "integration", "density evaluation budget exhausted");
    ++work.evaluations;
    const auto value = 4.0 * std::numbers::pi * Density(rho, r) * r * r;
    if (!std::isfinite(value)) throw Error("RadialGravity", "rho", "mass integrand is not finite");
    return value;
  }
  [[nodiscard]] static double Density(const RadialFunction& rho, double r) {
    const auto value = rho.Value(r);
    if (!(std::isfinite(value) && value >= 0.0))
      throw Error("RadialGravity", "rho", "density must be finite and non-negative");
    return value;
  }
  [[nodiscard]] Estimate Rule(const RadialFunction& rho, double lo, double hi, Work& work) const {
    const auto mid = std::midpoint(lo, hi), half = 0.5 * (hi - lo);
    auto kronrod = wk_[7] * Integrand(rho, mid, work);
    auto gauss = wg_[3] * Integrand(rho, mid, work);
    for (std::size_t j = 0; j < 7; ++j) {
      const auto pair = Integrand(rho, mid - half * x_[j], work) +
                        Integrand(rho, mid + half * x_[j], work);
      kronrod += wk_[j] * pair;
      if (j == 1) gauss += wg_[0] * pair;
      if (j == 3) gauss += wg_[1] * pair;
      if (j == 5) gauss += wg_[2] * pair;
    }
    const auto k = kronrod * half;
    return {k, std::abs(k - gauss * half)};
  }
  [[nodiscard]] double Integrate(const RadialFunction& rho, double lo, double hi,
                                 unsigned depth, Work& work) const {
    if (lo == hi) return 0.0;
    const auto estimate = Rule(rho, lo, hi, work);
    if (estimate.error <= settings_.absolute_tolerance +
                              settings_.relative_tolerance * std::abs(estimate.value))
      return estimate.value;
    if (depth == 0) throw Error("RadialGravity", "integration", "mass integral did not converge");
    const auto mid = std::midpoint(lo, hi);
    return Integrate(rho, lo, mid, depth - 1, work) + Integrate(rho, mid, hi, depth - 1, work);
  }
  void BuildInterval(LayerMass& layer, double lo, double hi, unsigned depth, Work& work) const {
    const auto estimate = Rule(layer.rho, lo, hi, work);
    if (estimate.error <= settings_.absolute_tolerance +
                              settings_.relative_tolerance * std::abs(estimate.value)) {
      const auto prefix = layer.pieces.empty()
          ? (layers_.empty() ? 0.0 : layers_.back().pieces.back().prefix + layers_.back().pieces.back().mass)
          : layer.pieces.back().prefix + layer.pieces.back().mass;
      layer.pieces.push_back({lo, hi, prefix, estimate.value});
      return;
    }
    if (depth >= settings_.max_depth)
      throw Error("RadialGravity", "integration", "mass integral did not converge at maximum depth");
    const auto mid = std::midpoint(lo, hi);
    BuildInterval(layer, lo, mid, depth + 1, work);
    BuildInterval(layer, mid, hi, depth + 1, work);
  }
  void CheckRadius(const LayerMass& layer, std::size_t i, double r) const {
    const auto [lo, hi] = layer.rho.Interval();
    if (!(std::isfinite(r) && r >= lo && r <= hi))
      throw Error("RadialGravity", std::format("layer {}", i), "radius lies outside the layer");
  }
  void CheckModelRadius(double r) const {
    if (!(std::isfinite(r) && r >= 0.0 && r <= layers_.back().rho.Interval().second))
      throw Error("RadialGravity", "radius", "must lie between the centre and model surface");
  }
  [[nodiscard]] std::size_t LayerAt(double r) const {
    auto i = std::size_t{0};
    while (i + 1 < layers_.size() && r >= layers_[i].rho.Interval().second) ++i;
    return i;
  }

  double G_;
  GravityIntegrationSettings settings_;
  std::vector<LayerMass> layers_;
};

}  // namespace planetmodel

#endif  // PLANETMODEL_GRAVITY_HPP
