#ifndef PLANETMODEL_RADIAL_HPP
#define PLANETMODEL_RADIAL_HPP

/// @file
/// Functions of the radius on one interval, with their first derivative.
///
/// A `RadialFunction` evaluates at any radius of its interval, and at one
/// within its tolerance outside, which is taken as the nearer end, as the
/// Python's fields take a radius within their rtol of an interval as on
/// it; beyond that it throws.  It is cheap to copy, the interpolant behind
/// it being shared, and its evaluation is `const` and thread-safe.  The
/// constructors here are the piecewise polynomials through knots that a
/// deck's columns are joined by: the not-a-knot and the natural cubic
/// splines, the parabola, and the polyline.

#include <format>
#include <functional>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Interpolation/CubicSpline.hpp>
#include <Interpolation/Linear.hpp>

#include "planetmodel/error.hpp"

namespace planetmodel {

/// A function of the radius on [lo, hi] and its first derivative.
class RadialFunction {
 public:
  /// What a function is built from: its value (`derivative` 0) or first
  /// derivative (1) at a radius already within the interval.
  using Evaluator = std::function<double(double r, int derivative)>;

  /// `evaluate` on [lo, hi], a radius within `tolerance` outside it taken
  /// as the nearer end.
  RadialFunction(double lo, double hi, double tolerance, Evaluator evaluate)
      : lo_(lo), hi_(hi), tolerance_(tolerance),
        evaluate_(std::make_shared<const Evaluator>(std::move(evaluate))) {}

  /// The value at `r`.
  [[nodiscard]] double Value(double r) const { return (*evaluate_)(Within(r), 0); }

  /// The first derivative at `r`.
  [[nodiscard]] double Derivative(double r) const { return (*evaluate_)(Within(r), 1); }

  /// The interval the function is defined on.
  [[nodiscard]] std::pair<double, double> Interval() const noexcept { return {lo_, hi_}; }

 private:
  double Within(double r) const {
    if (!(r >= lo_ - tolerance_ && r <= hi_ + tolerance_)) {
      throw Error("RadialFunction", "",
                  std::format("r = {} lies outside [{}, {}]", r, lo_, hi_));
    }
    return r < lo_ ? lo_ : (r > hi_ ? hi_ : r);
  }

  double lo_;
  double hi_;
  double tolerance_;
  std::shared_ptr<const Evaluator> evaluate_;
};

/// How a deck's knots are joined within a layer, the kinds both the Python
/// and the C++ read: `cubic` (scipy's not-a-knot `CubicSpline`, the
/// parabola through three knots), `natural` (no curvature at the ends) and
/// `linear`; two knots are a line under every kind.
enum class KnotKind { Cubic, Natural, Linear };

/// The piecewise polynomial of `kind` through knots (r, y), the radii
/// strictly increasing, on [r.front(), r.back()] with `tolerance`.
[[nodiscard]] inline RadialFunction ThroughKnots(std::vector<double> r, std::vector<double> y,
                                                 KnotKind kind, double tolerance) {
  if (r.size() != y.size() || r.size() < 2) {
    throw Error("ThroughKnots", "", "needs at least two knots and one value at each");
  }
  for (auto i = std::size_t{1}; i < r.size(); ++i) {
    if (!(r[i] > r[i - 1])) throw Error("ThroughKnots", "", "the knots must increase strictly");
  }
  const auto lo = r.front();
  const auto hi = r.back();
  // Each interpolant owns the knots it was given, shared by the copies of
  // the function.
  auto share = [](auto&& interpolant) {
    auto f = std::make_shared<const std::decay_t<decltype(interpolant)>>(
        std::forward<decltype(interpolant)>(interpolant));
    return RadialFunction::Evaluator([f](double x, int derivative) {
      return derivative == 0 ? f->template Evaluate<0>(x) : f->template Evaluate<1>(x);
    });
  };
  auto evaluator = RadialFunction::Evaluator{};
  if (r.size() == 2 || kind == KnotKind::Linear) {
    evaluator = share(Interpolation::Linear(std::move(r), std::move(y)));
  } else if (r.size() == 3 && kind == KnotKind::Cubic) {
    // the parabola in Newton's form, which no nearness to a knot upsets
    const auto d01 = (y[1] - y[0]) / (r[1] - r[0]);
    const auto d12 = (y[2] - y[1]) / (r[2] - r[1]);
    const auto d012 = (d12 - d01) / (r[2] - r[0]);
    evaluator = [x0 = r[0], x1 = r[1], y0 = y[0], d01, d012](double x, int derivative) {
      return derivative == 0 ? y0 + (x - x0) * (d01 + (x - x1) * d012)
                             : d01 + ((x - x0) + (x - x1)) * d012;
    };
  } else {
    const auto condition = kind == KnotKind::Cubic ? Interpolation::BoundaryCondition::NotAKnot
                                                   : Interpolation::BoundaryCondition::Natural;
    evaluator = share(Interpolation::CubicSpline(std::move(r), std::move(y), condition, 0.0, 0.0));
  }
  return RadialFunction(lo, hi, tolerance, std::move(evaluator));
}

}  // namespace planetmodel

#endif  // PLANETMODEL_RADIAL_HPP
