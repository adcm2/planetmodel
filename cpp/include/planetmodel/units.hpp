#ifndef PLANETMODEL_UNITS_HPP
#define PLANETMODEL_UNITS_HPP

/// @file
/// Physical dimensions, and the scales that remove them, as
/// `planetmodel.units` defines them.
///
/// **Dimensions** are integer exponents of (mass, length, time,
/// temperature), multiplied by adding them.  **Scales** say what one unit
/// of each base dimension is in SI, so that a quantity of dimensions d
/// stored under scales s relates to its SI value by
///
///     stored = SI / s.Factor(d),
///     s.Factor(d) = mass^d.mass length^d.length time^d.time
///                   temperature^d.temperature.
///
/// `Scales::Geophysical(a, rho)` takes the length a and the mass rho a^3
/// and fixes the time by G rho T^2 = 1, so that G is one in those units.
/// A model file states its scales and its constants, G among them, in its
/// own units; a code that wants SI multiplies by `Factor`.  The constants
/// and the formats of the unit strings are the Python's, restated here and
/// held equal to it by the fixtures.

#include <array>
#include <cmath>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "planetmodel/error.hpp"

namespace planetmodel::units {

/// The base dimensions, in the order of every exponent and scale.
inline constexpr auto BASE = std::array<std::string_view, 4>{
    "mass", "length", "time", "temperature"};

/// The gravitational constant, m^3 kg^-1 s^-2 (CODATA 2018).
inline constexpr auto G_SI = 6.6743e-11;

/// The conventional mean density of the Earth, kg m^-3: prescribed, not
/// computed.
inline constexpr auto EARTH_MEAN_DENSITY = 5515.0;

/// Physical dimensions as integer exponents of the base dimensions,
/// written with designated initialisers: `Dimensions{.mass = 1,
/// .length = -3}` is a density.
struct Dimensions {
  int mass = 0;
  int length = 0;
  int time = 0;
  int temperature = 0;

  friend constexpr bool operator==(const Dimensions&, const Dimensions&) = default;

  /// The dimensions of a product: the exponents added.
  friend constexpr Dimensions operator*(const Dimensions& a, const Dimensions& b) {
    return {.mass = a.mass + b.mass,
            .length = a.length + b.length,
            .time = a.time + b.time,
            .temperature = a.temperature + b.temperature};
  }

  /// The dimensions of a quotient: the exponents subtracted.
  friend constexpr Dimensions operator/(const Dimensions& a, const Dimensions& b) {
    return {.mass = a.mass - b.mass,
            .length = a.length - b.length,
            .time = a.time - b.time,
            .temperature = a.temperature - b.temperature};
  }

  /// The dimensions of the n-th power: every exponent times n.
  [[nodiscard]] constexpr Dimensions Pow(int n) const {
    return {.mass = mass * n,
            .length = length * n,
            .time = time * n,
            .temperature = temperature * n};
  }

  /// Whether every exponent is zero.
  [[nodiscard]] constexpr bool IsDimensionless() const {
    return mass == 0 && length == 0 && time == 0 && temperature == 0;
  }

  /// A udunits-style unit string: "kg m-3" in SI, "1" for a pure number,
  /// and "1" for every quantity when `si` is false, a non-dimensional
  /// model's numbers being pure whatever their dimensions.
  [[nodiscard]] std::string UnitString(bool si = true) const {
    if (!si) return "1";
    auto out = std::string{};
    const auto parts = std::array<std::pair<std::string_view, int>, 4>{
        {{"kg", mass}, {"m", length}, {"s", time}, {"K", temperature}}};
    for (const auto& [symbol, n] : parts) {
      if (n == 0) continue;
      if (!out.empty()) out += ' ';
      out += symbol;
      if (n != 1) out += std::to_string(n);
    }
    return out.empty() ? std::string{"1"} : out;
  }
};

inline constexpr auto DIMENSIONLESS = Dimensions{};
inline constexpr auto MASS = Dimensions{.mass = 1};
inline constexpr auto LENGTH = Dimensions{.length = 1};
inline constexpr auto TIME = Dimensions{.time = 1};
inline constexpr auto FREQUENCY = Dimensions{.time = -1};
inline constexpr auto DENSITY = Dimensions{.mass = 1, .length = -3};
inline constexpr auto VELOCITY = Dimensions{.length = 1, .time = -1};
inline constexpr auto GRAVITY = Dimensions{.length = 1, .time = -2};
inline constexpr auto MODULUS = Dimensions{.mass = 1, .length = -1, .time = -2};
inline constexpr auto VISCOSITY = Dimensions{.mass = 1, .length = -1, .time = -1};
inline constexpr auto GRAVITATIONAL_CONSTANT =
    Dimensions{.mass = -1, .length = 3, .time = -2};
inline constexpr auto TEMPERATURE = Dimensions{.temperature = 1};

/// The four values a `Scales` is built from, each one stored unit in SI,
/// named at the call: `Scales({.length = 6371e3, .mass = 6e24, .time =
/// 1e3})`, the temperature a kelvin unless given.
struct ScaleValues {
  double length = 1.0;
  double mass = 1.0;
  double time = 1.0;
  double temperature = 1.0;
};

/// What one unit of each base dimension is, in SI.  SI by default.
class Scales {
 public:
  /// The SI scales, every unit one.
  constexpr Scales() = default;

  /// Scales of the given values, each a positive finite number, or an
  /// `Error` naming the one that is not.
  explicit Scales(const ScaleValues& values) : values_(values) {
    const auto named = std::array<std::pair<std::string_view, double>, 4>{
        {{"length", values.length},
         {"mass", values.mass},
         {"time", values.time},
         {"temperature", values.temperature}}};
    for (const auto& [name, value] : named) {
      if (!(std::isfinite(value) && value > 0.0)) {
        throw Error("Scales", name,
                    std::format("a scale must be a positive finite number, got {}",
                                value));
      }
    }
  }

  /// The SI scales.
  [[nodiscard]] static constexpr Scales SI() { return Scales{}; }

  /// Scales in which G is one: `length` the body's radius in metres,
  /// `density` a prescribed reference density in kg m^-3, the mass
  /// density length^3, the time fixed by G density T^2 = 1, and
  /// `temperature` in kelvin.
  [[nodiscard]] static Scales Geophysical(double length,
                                          double density = EARTH_MEAN_DENSITY,
                                          double temperature = 1.0) {
    if (!(length > 0.0 && density > 0.0)) {
      throw Error("Scales", "Geophysical",
                  std::format("length and density must be positive, got {}, {}",
                              length, density));
    }
    return Scales({.length = length,
                   .mass = density * std::pow(length, 3.0),
                   .time = 1.0 / std::sqrt(G_SI * density),
                   .temperature = temperature});
  }

  [[nodiscard]] constexpr double Length() const { return values_.length; }
  [[nodiscard]] constexpr double Mass() const { return values_.mass; }
  [[nodiscard]] constexpr double Time() const { return values_.time; }
  [[nodiscard]] constexpr double Temperature() const { return values_.temperature; }

  /// Whether one stored unit of each base dimension is one SI unit.
  [[nodiscard]] constexpr bool IsSI() const {
    return values_.length == 1.0 && values_.mass == 1.0 && values_.time == 1.0 &&
           values_.temperature == 1.0;
  }

  /// The SI size of one stored unit of `dims`, in the Python's order of
  /// the base dimensions.
  [[nodiscard]] double Factor(const Dimensions& dims) const {
    return std::pow(values_.mass, dims.mass) * std::pow(values_.length, dims.length) *
           std::pow(values_.time, dims.time) *
           std::pow(values_.temperature, dims.temperature);
  }

  friend constexpr bool operator==(const Scales& a, const Scales& b) {
    return a.values_.length == b.values_.length && a.values_.mass == b.values_.mass &&
           a.values_.time == b.values_.time &&
           a.values_.temperature == b.values_.temperature;
  }

 private:
  ScaleValues values_{};
};

/// `dims->UnitString(si)`, or "unknown" where no dimensions are declared.
[[nodiscard]] inline std::string UnitString(const std::optional<Dimensions>& dims,
                                            bool si = true) {
  return dims ? dims->UnitString(si) : std::string{"unknown"};
}

/// The stored unit as a udunits scaled unit: the SI unit, preceded by the
/// size of one stored unit where the scales make it other than one
/// ("m s-1" in SI, "3865.300165 m s-1" under the Earth's geophysical
/// scales); "1" for a pure number and "unknown" where no dimensions are
/// declared.
[[nodiscard]] inline std::string ScaledUnit(const std::optional<Dimensions>& dims,
                                            const Scales& scales) {
  if (!dims) return "unknown";
  const auto si = dims->UnitString();
  const auto factor = scales.Factor(*dims);
  if (factor == 1.0) return si;
  return si == "1" ? std::format("{:.10g}", factor)
                   : std::format("{:.10g} {}", factor, si);
}

}  // namespace planetmodel::units

namespace planetmodel {

using units::Dimensions;
using units::EARTH_MEAN_DENSITY;
using units::G_SI;
using units::Scales;

}  // namespace planetmodel

#endif  // PLANETMODEL_UNITS_HPP
