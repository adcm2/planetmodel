#ifndef PLANETMODEL_RADIAL_MODEL_HPP
#define PLANETMODEL_RADIAL_MODEL_HPP

/**
 * @file
 * @brief An in-memory, layered radial material model and its MINEOS factory.
 *
 * The model owns immutable layer interpolants and derives material
 * properties at evaluation time so their derivatives follow the primitive
 * fields. The MINEOS factory supplies SI coordinates and values.
 */

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "planetmodel/deck.hpp"
#include "planetmodel/units.hpp"

namespace planetmodel {

/** @brief A scalar value and its radial derivative. */
struct RadialValue {
  double value = 0.0; /**< The property value. */
  double derivative = 0.0; /**< Its first radial derivative. */
};

/** @brief Layer-wide shear state, which quality factors do not determine. */
enum class ShearState { Unknown, Fluid, Solid };

/** @brief A named layer holding immutable radial fields. */
struct RadialLayer {
  std::optional<std::string> name{}; /**< MINEOS layer name, when known. */
  std::map<std::string, RadialFunction> fields{}; /**< Primitive fields on this layer. */
  ShearState shear_state = ShearState::Unknown; /**< Layer-wide classification, if known. */
};

/** @brief Sum two values and their derivatives. */
inline RadialValue operator+(RadialValue a, RadialValue b) {
  return {a.value + b.value, a.derivative + b.derivative};
}
/** @brief Subtract two values and their derivatives. */
inline RadialValue operator-(RadialValue a, RadialValue b) {
  return {a.value - b.value, a.derivative - b.derivative};
}
/** @brief Product rule for radial derivatives. */
inline RadialValue operator*(RadialValue a, RadialValue b) {
  return {a.value * b.value, a.derivative * b.value + a.value * b.derivative};
}
/** @brief Quotient rule for radial derivatives. */
inline RadialValue operator/(RadialValue a, RadialValue b) {
  return {a.value / b.value,
          (a.derivative * b.value - a.value * b.derivative) / (b.value * b.value)};
}
/** @brief Scale a value and its derivative. */
inline RadialValue operator*(double a, RadialValue b) {
  return {a * b.value, a * b.derivative};
}
/** @brief Add a constant without changing the derivative. */
inline RadialValue operator+(RadialValue a, double b) { return {a.value + b, a.derivative}; }
/** @brief Subtract a constant without changing the derivative. */
inline RadialValue operator-(RadialValue a, double b) { return {a.value - b, a.derivative}; }
namespace detail {
inline RadialValue Square(RadialValue a) { return a * a; }
}  // namespace detail

/** @brief One layer's material fields and pointwise elastic conversions. */
class RadialMaterial {
 public:
  /** @brief Build from fields that share a layer interval. */
  explicit RadialMaterial(std::map<std::string, RadialFunction> fields)
      : fields_(std::move(fields)) {
    for (const auto& [name, field] : fields_) {
      if (!interval_) interval_ = field.Interval();
      else if (*interval_ != field.Interval()) {
        throw Error("RadialMaterial", name, "fields must share one layer interval");
      }
    }
    if (!interval_) throw Error("RadialMaterial", "", "needs at least one field");
    if (HasAny({"A", "C", "F", "L", "N", "kappa", "mu"}) &&
        !HasAll({"A", "C", "F", "L", "N"}) && !HasAll({"kappa", "mu"}) &&
        !HasAll({"rho", "vpv", "vph", "vsv", "vsh", "eta"}) &&
        !HasAll({"rho", "vp", "vs"})) {
      throw Error("RadialMaterial", "", "incomplete elastic property set");
    }
    const bool any_velocity = HasAny({"vp", "vs", "vpv", "vph", "vsv", "vsh", "eta"});
    if (any_velocity && !Has("rho") && !HasAll({"A", "C", "F", "L", "N"}) &&
        !HasAll({"kappa", "mu"})) {
      throw Error("RadialMaterial", "rho", "velocity descriptions need density");
    }
    if (any_velocity && !HasAll({"rho", "vpv", "vph", "vsv", "vsh", "eta"}) &&
        !HasAll({"rho", "vp", "vs"}) && !HasAll({"A", "C", "F", "L", "N"}) &&
        !HasAll({"kappa", "mu"})) {
      throw Error("RadialMaterial", "", "incomplete elastic property set");
    }
  }

  /** @brief The primitive fields held by this material. */
  [[nodiscard]] const auto& Fields() const noexcept { return fields_; }
  /** @brief This material's radial interval. */
  [[nodiscard]] std::pair<double, double> Interval() const noexcept { return *interval_; }
  /** @brief Whether a primitive field is present. */
  [[nodiscard]] bool Has(std::string_view name) const { return fields_.contains(std::string{name}); }

  /** @brief A primitive or derived property and its radial derivative. */
  [[nodiscard]] RadialValue Evaluate(std::string_view name, double r) const {
    const auto description = Describe();
    if (Has(name) && Direct(name, description)) return Read(name, r);
    if (const auto value = Derived(name, r, description)) return *value;
    if (Has(name)) return Read(name, r);
    throw Error("RadialMaterial", name, "property is unavailable from the fields in this layer");
  }

  /** @brief A value/derivative field safe to copy and evaluate independently. */
  [[nodiscard]] RadialFunction Property(std::string name) const {
    const auto [lo, hi] = *interval_;
    const auto tol = DEFAULT_RTOL * (hi - lo);
    const auto material = *this;
    return RadialFunction(lo, hi, tol, [material, name = std::move(name)](double r, int d) {
      const auto x = material.Evaluate(name, r);
      return d == 0 ? x.value : x.derivative;
    });
  }

  /** @brief Whether every held shear-bearing field vanishes at r. */
  [[nodiscard]] bool IsFluid(double r) const {
    auto found = false;
    for (const auto name : {"vs", "vsv", "vsh", "mu", "L", "N"}) {
      if (!Has(name)) continue;
      found = true;
      if (Read(name, r).value != 0.0) return false;
    }
    if (!found) throw Error("RadialMaterial", "", "fluidity needs shear velocities or moduli");
    return true;
  }

 private:
  enum class Description { None, Love, IsotropicModuli, VtiVelocities, IsotropicVelocities };

  [[nodiscard]] bool HasAny(std::initializer_list<std::string_view> names) const {
    for (const auto name : names) if (Has(name)) return true;
    return false;
  }
  [[nodiscard]] bool HasAll(std::initializer_list<std::string_view> names) const {
    for (const auto name : names) if (!Has(name)) return false;
    return true;
  }
  [[nodiscard]] Description Describe() const {
    if (HasAll({"A", "C", "F", "L", "N"})) return Description::Love;
    if (HasAll({"kappa", "mu"})) return Description::IsotropicModuli;
    if (HasAll({"rho", "vpv", "vph", "vsv", "vsh", "eta"}))
      return Description::VtiVelocities;
    if (HasAll({"rho", "vp", "vs"})) return Description::IsotropicVelocities;
    return Description::None;
  }
  [[nodiscard]] static bool In(std::string_view name,
                               std::initializer_list<std::string_view> names) {
    for (const auto item : names) if (name == item) return true;
    return false;
  }
  [[nodiscard]] bool Direct(std::string_view name, Description description) const {
    if (name == "rho") return true;
    if (In(name, {"A", "C", "F", "L", "N"})) return description == Description::Love;
    if (In(name, {"kappa", "mu"})) return description == Description::IsotropicModuli;
    if (In(name, {"vpv", "vph", "vsv", "vsh", "eta"}))
      return description == Description::VtiVelocities;
    if (In(name, {"vp", "vs"})) return description == Description::IsotropicVelocities;
    return true;
  }
  [[nodiscard]] RadialValue Read(std::string_view name, double r) const {
    const auto& f = fields_.at(std::string{name});
    return {f.Value(r), f.Derivative(r)};
  }
  [[nodiscard]] std::optional<RadialValue> Derived(std::string_view name, double r,
                                                    Description description) const {
    const auto four = RadialValue{4.0, 0.0}, five = RadialValue{5.0, 0.0};
    const auto nine = RadialValue{9.0, 0.0}, fifteen = RadialValue{15.0, 0.0};
    RadialValue A{}, C{}, F{}, L{}, N{}, kappa{}, mu{};
    if (description == Description::Love) {
      A = Read("A", r); C = Read("C", r); F = Read("F", r);
      L = Read("L", r); N = Read("N", r);
      kappa = (four * A + C + four * F - four * N) / nine;
      mu = (A + C - 2.0 * F + five * N + 6.0 * L) / fifteen;
      if (name == "kappa") return kappa;
      if (name == "mu") return mu;
      return std::nullopt;
    }
    if (description == Description::IsotropicModuli) {
      kappa = Read("kappa", r); mu = Read("mu", r);
      A = kappa + (4.0 / 3.0) * mu; C = A; F = kappa - (2.0 / 3.0) * mu;
      L = mu; N = mu;
    } else if (description == Description::VtiVelocities) {
      const auto rho = Read("rho", r), vpv = Read("vpv", r), vph = Read("vph", r);
      const auto vsv = Read("vsv", r), vsh = Read("vsh", r), eta = Read("eta", r);
      A = rho * detail::Square(vph); C = rho * detail::Square(vpv);
      L = rho * detail::Square(vsv); N = rho * detail::Square(vsh);
      F = eta * (A - 2.0 * L);
      kappa = (four * A + C + four * F - four * N) / nine;
      mu = (A + C - 2.0 * F + five * N + 6.0 * L) / fifteen;
    } else if (description == Description::IsotropicVelocities) {
      const auto rho = Read("rho", r), vp = Read("vp", r), vs = Read("vs", r);
      mu = rho * detail::Square(vs);
      kappa = rho * (detail::Square(vp) - (4.0 / 3.0) * detail::Square(vs));
      A = kappa + (4.0 / 3.0) * mu; C = A;
      F = kappa - (2.0 / 3.0) * mu; L = mu; N = mu;
    } else {
      return std::nullopt;
    }
    if (name == "A") return A;
    if (name == "C") return C;
    if (name == "F") return F;
    if (name == "L") return L;
    if (name == "N") return N;
    if (name == "kappa") return kappa;
    if (name == "mu") return mu;
    return std::nullopt;
  }
  std::map<std::string, RadialFunction> fields_;
  std::optional<std::pair<double, double>> interval_;
};

/** @brief A layered radial model with its units and reference metadata. */
class RadialModel {
 public:
  /** @brief Build an in-memory model; explicit G is in model units. */
  RadialModel(Skeleton skeleton, std::vector<RadialLayer> layers, Scales scales = Scales::SI(),
              std::optional<double> G = {}, std::optional<double> omega_ref = {}, std::string input = {},
              std::vector<std::optional<std::string>> interfaces = {},
              std::vector<std::vector<double>> breakpoints = {})
      : skeleton_(std::move(skeleton)), layers_(std::move(layers)), scales_(scales),
        G_(G.value_or(G_SI / scales.Factor(units::GRAVITATIONAL_CONSTANT))),
        omega_ref_(omega_ref), input_(std::move(input)), interfaces_(std::move(interfaces)),
        breakpoints_(std::move(breakpoints)) {
    if (layers_.size() != skeleton_.NLayers()) throw Error("RadialModel", "layers", "count differs from skeleton");
    if (!(std::isfinite(G_) && G_ > 0.0)) throw Error("RadialModel", "G", "must be positive and finite");
    if (omega_ref_ && !(std::isfinite(*omega_ref_) && *omega_ref_ > 0.0))
      throw Error("RadialModel", "omega_ref", "must be positive and finite");
    if (!interfaces_.empty() && interfaces_.size() !=
        skeleton_.NLayers() + (skeleton_.IsHollow() ? 1U : 0U))
      throw Error("RadialModel", "interfaces", "count differs from skeleton boundaries");
    if (breakpoints_.empty()) {
      breakpoints_.reserve(layers_.size());
      for (std::size_t i = 0; i < layers_.size(); ++i) {
        const auto [lo, hi] = skeleton_.Interval(i);
        breakpoints_.push_back({lo, hi});
      }
    }
    if (breakpoints_.size() != layers_.size())
      throw Error("RadialModel", "breakpoints", "count differs from layers");
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      const auto want = skeleton_.Interval(i);
      const auto& points = breakpoints_[i];
      if (points.size() < 2 || points.front() != want.first || points.back() != want.second)
        throw Error("RadialModel", "breakpoints", "each layer must include its two endpoints");
      for (std::size_t j = 0; j < points.size(); ++j) {
        if (!std::isfinite(points[j]) || (j > 0 && !(points[j] > points[j - 1])) ||
            points[j] < want.first || points[j] > want.second)
          throw Error("RadialModel", "breakpoints", "must increase strictly within the layer");
      }
      for (const auto& [name, field] : layers_[i].fields) {
        if (field.Interval() != want) throw Error("RadialModel", name, "field interval differs from its layer");
      }
    }
  }

  /** @brief Layer boundaries. */
  [[nodiscard]] const Skeleton& GetSkeleton() const noexcept { return skeleton_; }
  /** @brief Every layer, counted from the centre outward. */
  [[nodiscard]] const std::vector<RadialLayer>& Layers() const noexcept { return layers_; }
  /** @brief Layer i. */
  [[nodiscard]] const RadialLayer& Layer(std::size_t i) const { return layers_.at(i); }
  /** @brief The scales that convert model units to SI. */
  [[nodiscard]] Scales GetScales() const noexcept { return scales_; }
  /** @brief The gravitational constant in the model's units. */
  [[nodiscard]] double G() const noexcept { return G_; }
  /** @brief Reference angular frequency in model units, when the input supplies one. */
  [[nodiscard]] std::optional<double> OmegaRef() const noexcept { return omega_ref_; }
  /** @brief Input identity, usually the deck path or source label. */
  [[nodiscard]] const std::string& Input() const noexcept { return input_; }
  /** @brief Named MINEOS interfaces, excluding the centre of a full body. */
  [[nodiscard]] const auto& InterfaceNames() const noexcept { return interfaces_; }
  /** @brief Source knots within layer i; endpoints for a constructed smooth layer. */
  [[nodiscard]] const std::vector<double>& IntegrationBreakpoints(std::size_t i) const {
    return breakpoints_.at(i);
  }
  /** @brief Material fields and conversions for layer i. */
  [[nodiscard]] RadialMaterial Material(std::size_t i) const { return RadialMaterial(layers_.at(i).fields); }
  /** @brief Layer-wide fluid status; unknown layers are refused. */
  [[nodiscard]] bool IsFluid(std::size_t i) const {
    const auto state = layers_.at(i).shear_state;
    if (state == ShearState::Unknown)
      throw Error("RadialModel", "fluidity", "layer shear state is unknown");
    return state == ShearState::Fluid;
  }

 private:
  Skeleton skeleton_;
  std::vector<RadialLayer> layers_;
  Scales scales_;
  double G_;
  std::optional<double> omega_ref_;
  std::string input_;
  std::vector<std::optional<std::string>> interfaces_;
  std::vector<std::vector<double>> breakpoints_;
};

/** @brief Build from a MINEOS deck, whose radii and fields are in SI. */
[[nodiscard]] inline RadialModel MineosModel(const Deck& deck, std::string_view kind = "cubic") {
  const auto functions = DeckLayers(deck, kind);
  const auto names = MineosNames(deck);
  for (const auto qname : {"qkappa", "qmu"}) {
    if (!deck.HasColumn(qname)) continue;
    for (const auto q : deck.Column(qname)) {
      if (!(std::isfinite(q) && q >= 0.0))
        throw Error(deck.Source(), qname, "quality factors must be finite and non-negative");
    }
  }
  auto layers = std::vector<RadialLayer>{};
  auto breakpoints = std::vector<std::vector<double>>{};
  const auto slices = deck.Layers();
  breakpoints.reserve(slices.size());
  for (const auto& [begin, end] : slices) {
    breakpoints.emplace_back(deck.Radius().begin() + static_cast<std::ptrdiff_t>(begin),
                             deck.Radius().begin() + static_cast<std::ptrdiff_t>(end));
  }
  layers.reserve(functions.layers.size());
  for (std::size_t i = 0; i < functions.layers.size(); ++i) {
    auto fields = functions.layers[i];
    auto fill = [&](std::string_view name, double value) {
      if (fields.contains(std::string{name})) return;
      const auto [lo, hi] = functions.skeleton.Interval(i);
      fields.emplace(std::string{name}, RadialFunction(lo, hi, DEFAULT_RTOL * functions.skeleton.Span(),
          [value](double, int derivative) { return derivative == 0 ? value : 0.0; }));
    };
    if (!fields.contains("vp")) {
      if (!functions.layers[i].contains("vph")) {
        const auto vpv = functions.layers[i].at("vpv");
        const auto [lo, hi] = functions.skeleton.Interval(i);
        fields.insert_or_assign("vph", RadialFunction(lo, hi, DEFAULT_RTOL * functions.skeleton.Span(),
          [vpv](double r, int d) { return d == 0 ? vpv.Value(r) : vpv.Derivative(r); }));
      }
      if (!functions.layers[i].contains("vsh")) {
        const auto vsv = functions.layers[i].at("vsv");
        const auto [lo, hi] = functions.skeleton.Interval(i);
        fields.insert_or_assign("vsh", RadialFunction(lo, hi, DEFAULT_RTOL * functions.skeleton.Span(),
          [vsv](double r, int d) { return d == 0 ? vsv.Value(r) : vsv.Derivative(r); }));
      }
      if (!functions.layers[i].contains("eta")) fill("eta", 1.0);
    }
    const auto material = RadialMaterial(fields);
    const auto all_shear_zero = std::ranges::all_of(breakpoints[i],
        [&](double r) { return material.IsFluid(r); });
    layers.push_back({.name = names.layers[i], .fields = std::move(fields),
                      .shear_state = all_shear_zero ? ShearState::Fluid : ShearState::Solid});
  }
  const auto header = ReadMineosHeader(deck.Header(), deck.Source());
  const auto omega = std::isfinite(header.tref) && header.tref > 0.0
                         ? std::optional{2.0 * std::acos(-1.0) / header.tref}
                         : std::nullopt;
  return RadialModel(functions.skeleton, std::move(layers), Scales::SI(), G_SI, omega,
                     deck.Source(), names.interfaces, std::move(breakpoints));
}

/** @brief Read a MINEOS deck and build its in-memory radial model. */
[[nodiscard]] inline RadialModel ReadMineosModel(const std::filesystem::path& path,
                                                 std::string_view kind = "cubic") {
  return MineosModel(ReadDeck(path, MINEOS()), kind);
}

}  // namespace planetmodel

#endif  // PLANETMODEL_RADIAL_MODEL_HPP
