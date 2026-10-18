#ifndef PLANETMODEL_MODELFILE_HPP
#define PLANETMODEL_MODELFILE_HPP

/**
 * @file
 * @brief The model file, as `planetmodel.modelfile` defines it and the
 * format page describes it: a model in one netCDF-4 file, on its
 * reference skeleton.
 *
 * `ReadModelFile(path)` reads a file whole and holds it to the structure
 * the Python's `validate_structure` checks, refusing the first fault with
 * an `Error` naming the file and the place; the numbers are kept as
 * stored, in the file's scales.  `ModelFile::FieldOn(layer, name)` builds a
 * field of one variable on one layer, which evaluates anywhere on its
 * layer by the rules the file states: a radial variable, or a layer a
 * lateral variable names in `radial_layers`, as a function of the radius;
 * a lateral one by its radial and lateral rules, its components of rank
 * one and above interpolated in Cartesian components; a reference with a
 * perturbation by the perturbation's kind.  `ModelFile::Mapping()` gives
 * the mapping, from the boundary displacements of the `geometry` group or
 * from the `displacement` field.
 *
 * A field holds the interpolants it was built with and nothing mutable,
 * so it is cheap to copy and safe to evaluate from many threads; the
 * `ModelFile` holds only numbers.  A code reads the file, builds the
 * fields it wants, samples them onto its own discretisation and lets them
 * go:
 *
 * @code
 * const auto file = planetmodel::ReadModelFile("earth.nc");
 * const auto vs = file.FieldOn(3, "vsv").AsScalar();
 * const auto value = vs.Value(r, theta, phi);
 * const auto u = file.Mapping().Displacement({x, y, z});
 * @endcode
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "planetmodel/character.hpp"
#include "planetmodel/error.hpp"
#include "planetmodel/frames.hpp"
#include "planetmodel/gll.hpp"
#include "planetmodel/grid.hpp"
#include "planetmodel/netcdf.hpp"
#include "planetmodel/rules.hpp"
#include "planetmodel/skeleton.hpp"
#include "planetmodel/units.hpp"

namespace planetmodel {

/** @brief The schema this reader reads; bumped only for an incompatible change. */
inline constexpr std::string_view MODEL_SCHEMA = "planetmodel.model/1";

/** @brief How a perturbation combines with its reference. */
inline constexpr auto PERTURBATION_KINDS =
    std::array<std::string_view, 3>{"absolute", "relative", "log"};

/** @brief The frame components are given in. */
enum class Frame {
  Spherical, /**< The local frame (e_r, e_theta, e_phi) at the point. */
  Cartesian  /**< The Cartesian frame of the coordinates. */
};

namespace detail {

/// The refusal of a file, naming the place of the fault.
[[noreturn]] inline void Fail(const std::string& source, std::string_view where,
                              std::string_view message) {
  throw Error(source, where, message);
}

/// An attribute's text, or empty where it is absent or a number.
inline std::optional<std::string> TextOf(const netcdf::Attributes& attrs, std::string_view key) {
  const auto it = attrs.find(std::string{key});
  if (it == attrs.end() || !std::holds_alternative<std::string>(it->second)) return std::nullopt;
  return std::get<std::string>(it->second);
}

/// An attribute's numbers, or empty where it is absent or text.
inline std::optional<std::vector<double>> NumbersOf(const netcdf::Attributes& attrs,
                                                   std::string_view key) {
  const auto it = attrs.find(std::string{key});
  if (it == attrs.end() || !std::holds_alternative<std::vector<double>>(it->second)) {
    return std::nullopt;
  }
  return std::get<std::vector<double>>(it->second);
}

/// An attribute's one number, or empty where it is absent, text or many.
inline std::optional<double> NumberOf(const netcdf::Attributes& attrs, std::string_view key) {
  const auto v = NumbersOf(attrs, key);
  return v && v->size() == 1 ? std::optional<double>{(*v)[0]} : std::nullopt;
}

/// Four exponents as dimensions, in the order `units.BASE` gives them.
inline std::optional<Dimensions> DimensionsFrom(const std::optional<std::vector<double>>& d) {
  if (!d || d->size() != 4) return std::nullopt;
  return Dimensions{.mass = static_cast<int>((*d)[0]),
                    .length = static_cast<int>((*d)[1]),
                    .time = static_cast<int>((*d)[2]),
                    .temperature = static_cast<int>((*d)[3])};
}

}  // namespace detail

/** @brief One variable of a part, its values as stored. */
struct FileVariable {
  std::string name{};                    /**< The variable's name. */
  std::vector<std::string> dimensions{}; /**< Its dimensions' names. */
  std::vector<std::size_t> shape{};      /**< Its dimensions' lengths. */
  std::vector<double> values{};          /**< Its values, C order, NaN for its fill. */
  netcdf::Attributes attributes{};       /**< Its attributes. */

  /**
   * @brief A text attribute.
   * @param key The attribute.
   * @return Its text, or empty where there is no such text attribute.
   */
  [[nodiscard]] std::optional<std::string> Text(std::string_view key) const {
    return detail::TextOf(attributes, key);
  }

  /**
   * @brief A numeric attribute.
   * @param key The attribute.
   * @return Its numbers, or empty where there is no such numeric attribute.
   */
  [[nodiscard]] std::optional<std::vector<double>> Numbers(std::string_view key) const {
    return detail::NumbersOf(attributes, key);
  }

  /** @brief Whether the variable varies with direction. */
  [[nodiscard]] bool IsLateral() const {
    return dimensions.size() >= 3 && dimensions[1] == "latitude" &&
           dimensions[2] == "longitude";
  }
};

/** @brief One part of a model file: a run of layers on one grid. */
struct FilePart {
  int index = 0;                         /**< `part_<index>`. */
  std::vector<int> layers{};             /**< The layers it holds, contiguous. */
  std::size_t ngll = 0;                  /**< The nodes per element. */
  std::vector<int> element_layer{};      /**< The layer of every element. */
  std::vector<double> element_bounds{};  /**< Each element's ends, two per element. */
  std::vector<double> radius{};          /**< Every node's radius, element by element. */
  std::optional<AngularGrid> grid{};     /**< The angular grid. */
  std::optional<std::vector<double>> colatitude_weights{}; /**< Where the grid has them. */
  std::map<std::string, FileVariable> variables{};          /**< The variables by name. */

  /**
   * @brief The elements of a layer of the part, as [begin, end).
   * @param layer The layer.
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> Elements(int layer) const {
    const auto first = std::ranges::find(element_layer, layer) - element_layer.begin();
    const auto last = std::ranges::find_last(element_layer, layer).begin() -
                      element_layer.begin();
    return {static_cast<std::size_t>(first), static_cast<std::size_t>(last) + 1};
  }
};

/**
 * @brief The `geometry` group: the mapping as displacements of surfaces,
 * linear in r between them.
 */
struct FileGeometry {
  std::vector<double> radii{};          /**< The surfaces' radii, increasing. */
  std::optional<AngularGrid> grid{};    /**< The grid the displacements are on. */
  std::vector<double> values{};         /**< (boundary, ntheta, nphi), C order. */
  std::optional<std::vector<double>> poles{}; /**< (boundary, 2), where the grid lacks a pole. */
  std::string lateral{};                /**< The lateral rule stated. */
  std::string between{};                /**< The rule between surfaces: `linear`. */
};

/** @brief A constant of the model, in its units. */
struct FileConstant {
  double value = 0.0;                     /**< In the file's scales. */
  std::optional<Dimensions> dimensions{}; /**< Where the file states them. */
  std::string meaning{};                  /**< What it is, where the file says. */
};

class ModelFile;
class ScalarField;

/**
 * @brief A variable on one layer as a function of the point, by the rules
 * the file states.
 */
class Field {
 public:
  /** @brief The field's name. */
  [[nodiscard]] const std::string& Name() const noexcept { return name_; }
  /** @brief The layer it is on. */
  [[nodiscard]] int Layer() const noexcept { return layer_; }
  /** @brief The layer's radii. */
  [[nodiscard]] std::pair<double, double> Interval() const noexcept { return interval_; }
  /** @brief The field's character, its law under a mapping. */
  [[nodiscard]] const Character& GetCharacter() const noexcept { return character_; }
  /** @brief The number of stored components. */
  [[nodiscard]] std::size_t Size() const { return character_.StoredSize(); }
  /** @brief Its dimensions, where the file states them. */
  [[nodiscard]] const std::optional<Dimensions>& GetDimensions() const noexcept {
    return dimensions_;
  }
  /** @brief Whether it depends on the radius alone on its layer. */
  [[nodiscard]] bool IsRadial() const noexcept {
    return reference_->radial && (!perturbation_ || perturbation_->radial);
  }
  /** @brief The kind of its perturbation, where it is a perturbed field. */
  [[nodiscard]] const std::optional<std::string>& PerturbationKind() const noexcept {
    return kind_;
  }

  /**
   * @brief The stored components at a point.
   * @param r The radius, within the layer to the file's tolerance.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @param frame The frame the components are given in.
   * @return The components, `Size()` of them.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::vector<double> Evaluate(double r, double theta, double phi,
                                             Frame frame) const {
    auto out = Read(*reference_, r, theta, phi, frame).value;
    if (perturbation_) {
      const auto delta = Read(*perturbation_, r, theta, phi, frame).value[0];
      out[0] = *kind_ == "absolute"   ? out[0] + delta
               : *kind_ == "relative" ? out[0] * (1.0 + delta)
                                      : out[0] * std::exp(delta);
    }
    return out;
  }

  /**
   * @brief The stored components' derivatives in r, theta and phi at a
   * point: of the Cartesian components for a field of rank one and above.
   * @param r The radius, within the layer to the file's tolerance.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @return The derivatives in r, theta and phi, `Size()` of each.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::array<std::vector<double>, 3> Derivatives(double r, double theta,
                                                               double phi) const {
    auto ref = Read(*reference_, r, theta, phi, Frame::Cartesian, true);
    if (!perturbation_) return std::move(ref.derivative);
    // a perturbed field is of rank 0, its perturbation too
    const auto dp = Read(*perturbation_, r, theta, phi, Frame::Cartesian, true);
    const auto value = ref.value[0];
    const auto delta = dp.value[0];
    for (auto d = std::size_t{0}; d < 3; ++d) {
      const auto d_ref = ref.derivative[d][0];
      const auto d_delta = dp.derivative[d][0];
      ref.derivative[d][0] = *kind_ == "absolute" ? d_ref + d_delta
                             : *kind_ == "relative"
                                 ? d_ref * (1.0 + delta) + value * d_delta
                                 : std::exp(delta) * (d_ref + value * d_delta);
    }
    return std::move(ref.derivative);
  }

  /**
   * @brief The field read as a number, for a field of rank zero.
   * @throws planetmodel::Error if the field's rank is not zero.
   */
  [[nodiscard]] ScalarField AsScalar() const;

 private:
  friend class ModelFile;

  /// What one variable is on one layer, at the nodes of its elements: the
  /// stored components in the spherical frame where it is radial, or the
  /// lateral rule's interpolants of the Cartesian components.
  struct Piece {
    bool radial = true;
    Character character{0};
    RadialRule rule = RadialRule::Polynomial;
    std::optional<RadialLayout> layout{};
    std::vector<double> nodal{};               // [(node * ncomp) + c], radial
    std::vector<LateralInterpolant> lateral{};  // [(node * ncomp) + c], lateral
  };

  struct Reading {
    std::vector<double> value{};
    std::array<std::vector<double>, 3> derivative{};
  };

  Field() = default;

  /// A piece's components at a point in `frame`, and with `derivatives`
  /// their Cartesian derivatives.
  static Reading Read(const Piece& p, double r, double theta, double phi, Frame frame,
                      bool derivatives = false) {
    const auto n = p.character.StoredSize();
    const auto rank = p.character.Rank();
    const auto voigt = p.character.Voigt();
    auto out = Reading{};
    out.value.assign(n, 0.0);
    if (derivatives) {
      for (auto& d : out.derivative) d.assign(n, 0.0);
    }
    // the element's radial weights, once for every component
    const auto& layout = *p.layout;
    const auto e = layout.ElementOf(r);
    const auto ngll = layout.Ngll();
    auto w = std::vector<double>(ngll);
    auto dw = std::vector<double>(derivatives ? ngll : 0);
    layout.Weights(e, r, p.rule, w, dw);
    if (p.radial) {
      auto sph = std::vector<double>(n, 0.0);
      auto dsph = std::vector<double>(n, 0.0);
      for (auto k = std::size_t{0}; k < ngll; ++k) {
        const auto* at = &p.nodal[(e * ngll + k) * n];
        for (auto c = std::size_t{0}; c < n; ++c) {
          sph[c] += w[k] * at[c];
          if (derivatives) dsph[c] += dw[k] * at[c];
        }
      }
      if (rank == 0 || frame == Frame::Spherical) out.value = sph;
      const auto R = SphericalFrame(theta, phi);
      const auto Rs = std::vector<Matrix3>(static_cast<std::size_t>(rank), R);
      if (rank > 0 && frame == Frame::Cartesian) out.value = RotateStored(sph, rank, voigt, Rs);
      if (!derivatives) return out;
      out.derivative[0] = rank == 0 ? dsph : RotateStored(dsph, rank, voigt, Rs);
      if (rank == 0) return out;
      // the frame's own derivatives, one factor at a time
      const auto dR = SphericalFrameDerivatives(theta, phi);
      for (auto d = 0; d < 2; ++d) {
        for (auto s = 0; s < rank; ++s) {
          auto Ms = Rs;
          Ms[static_cast<std::size_t>(s)] = dR[static_cast<std::size_t>(d)];
          const auto term = RotateStored(sph, rank, voigt, Ms);
          for (auto c = std::size_t{0}; c < n; ++c) {
            out.derivative[static_cast<std::size_t>(d) + 1][c] += term[c];
          }
        }
      }
      return out;
    }
    auto cart = std::vector<double>(n, 0.0);
    for (auto k = std::size_t{0}; k < ngll; ++k) {
      for (auto c = std::size_t{0}; c < n; ++c) {
        const auto& f = p.lateral[((e * ngll) + k) * n + c];
        const auto v = f.Value(theta, phi);
        cart[c] += w[k] * v;
        if (derivatives) {
          const auto [gt, gp] = f.Gradient(theta, phi);
          out.derivative[0][c] += dw[k] * v;
          out.derivative[1][c] += w[k] * gt;
          out.derivative[2][c] += w[k] * gp;
        }
      }
    }
    if (rank == 0 || frame == Frame::Cartesian) {
      out.value = cart;
    } else {
      const auto R = SphericalFrame(theta, phi);
      auto Rt = Matrix3{};
      for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
          Rt[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
              R[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
        }
      }
      out.value = RotateStored(cart, rank, voigt,
                               std::vector<Matrix3>(static_cast<std::size_t>(rank), Rt));
    }
    return out;
  }

  std::string source_{};
  std::string name_{};
  int layer_ = 0;
  std::pair<double, double> interval_{};
  Character character_{0};
  std::optional<Dimensions> dimensions_{};
  std::shared_ptr<const Piece> reference_{};
  std::shared_ptr<const Piece> perturbation_{};
  std::optional<std::string> kind_{};
};

/** @brief A field of rank zero, read as a number. */
class ScalarField {
 public:
  /**
   * @brief The value at a point.
   * @param r The radius, within the layer to the file's tolerance.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] double Value(double r, double theta, double phi) const {
    return field_.Evaluate(r, theta, phi, Frame::Spherical)[0];
  }

  /**
   * @brief The derivatives in r, theta and phi at a point.
   * @param r The radius, within the layer to the file's tolerance.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @return (d/dr, d/dtheta, d/dphi).
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::array<double, 3> Gradient(double r, double theta, double phi) const {
    const auto d = field_.Derivatives(r, theta, phi);
    return {d[0][0], d[1][0], d[2][0]};
  }

  /** @brief The field it reads. */
  [[nodiscard]] const Field& Underlying() const noexcept { return field_; }

 private:
  friend class Field;
  explicit ScalarField(Field field) : field_(std::move(field)) {}
  Field field_;
};

/**
 * @brief The field read as a number, for a field of rank zero.
 * @throws planetmodel::Error if the field's rank is not zero.
 */
inline ScalarField Field::AsScalar() const {
  if (character_.Rank() != 0) {
    throw Error(source_, name_,
                std::format("is of rank {}, not a scalar: read its components with Evaluate",
                            character_.Rank()));
  }
  return ScalarField(*this);
}

/**
 * @brief The mapping a model file holds: m(X) = X + u(X), from the
 * displacements of surfaces or from a displacement field, or the
 * identity.
 */
class ModelMapping {
 public:
  /** @brief The form the file holds the mapping in. */
  enum class Form {
    Identity, /**< The file holds no mapping. */
    Stack,    /**< The `geometry` group: surfaces' displacements, linear in r between. */
    Field     /**< The `displacement` field of every part. */
  };

  /** @brief The form the mapping is held in. */
  [[nodiscard]] Form GetForm() const noexcept { return form_; }

  /**
   * @brief The displacement u(X) = m(X) - X, Cartesian, at a reference
   * point.
   * @param X The point, Cartesian, within the reference body to the
   *        file's tolerance.
   * @throws planetmodel::Error if the point lies outside the body.
   */
  [[nodiscard]] Vector3 Displacement(const Vector3& X) const {
    const auto r = std::hypot(X[0], X[1], X[2]);
    if (!(r >= inner_ - tolerance_ && r <= outer_ + tolerance_)) {
      throw Error(source_, "mapping",
                  std::format("|X| = {} lies outside the body's [{}, {}]", r, inner_, outer_));
    }
    if (form_ == Form::Identity) return {0.0, 0.0, 0.0};
    const auto theta = std::atan2(std::hypot(X[0], X[1]), X[2]);
    const auto phi = std::atan2(X[1], X[0]);
    if (form_ == Form::Field) {
      const auto u =
          fields_[LayerOf(r)].Evaluate(std::clamp(r, inner_, outer_), theta, phi, Frame::Cartesian);
      return {u[0], u[1], u[2]};
    }
    // a radial stretch: h along e_r, X h / r, the limit dh/dr X at the centre
    const auto i = StackLayer(r);
    const auto lo = radii_[i];
    const auto hi = radii_[i + 1];
    const auto below = Relief(i, theta, phi);
    const auto above = Relief(i + 1, theta, phi);
    const auto t = (r - lo) / (hi - lo);
    const auto h = (1.0 - t) * below + t * above;
    const auto ratio = r > R_FLOOR * outer_ ? h / r : (above - below) / (hi - lo);
    return {X[0] * ratio, X[1] * ratio, X[2] * ratio};
  }

  /**
   * @brief The mapped point m(X) = X + u(X), Cartesian.
   * @param X The reference point, Cartesian, within the body.
   * @throws planetmodel::Error if the point lies outside the body.
   */
  [[nodiscard]] Vector3 operator()(const Vector3& X) const {
    const auto u = Displacement(X);
    return {X[0] + u[0], X[1] + u[1], X[2] + u[2]};
  }

 private:
  friend class ModelFile;

  /// A point closer to the origin than this fraction of the outer radius
  /// is the origin, as `mapping.R_FLOOR_FRACTION` takes it.
  static constexpr auto R_FLOOR = 1e-9;

  std::size_t LayerOf(double r) const {
    const auto i = std::upper_bound(boundaries_.begin(), boundaries_.end(), r) -
                   boundaries_.begin() - 1;
    return static_cast<std::size_t>(
        std::clamp<std::ptrdiff_t>(i, 0, static_cast<std::ptrdiff_t>(boundaries_.size()) - 2));
  }

  std::size_t StackLayer(double r) const {
    const auto i = std::upper_bound(radii_.begin(), radii_.end(), r) - radii_.begin() - 1;
    return static_cast<std::size_t>(
        std::clamp<std::ptrdiff_t>(i, 0, static_cast<std::ptrdiff_t>(radii_.size()) - 2));
  }

  double Relief(std::size_t k, double theta, double phi) const {
    const auto& f = reliefs_[k];
    return f ? f->Value(theta, phi) : 0.0;
  }

  Form form_ = Form::Identity;
  std::string source_{};
  double inner_ = 0.0;
  double outer_ = 0.0;
  double tolerance_ = 0.0;
  std::vector<double> boundaries_{};
  std::vector<Field> fields_{};
  std::vector<double> radii_{};
  std::vector<std::optional<LateralInterpolant>> reliefs_{};
};

/** @brief A model file, read and checked, its numbers as stored. */
class ModelFile {
 public:
  /** @brief The path it was read from. */
  [[nodiscard]] const std::string& Source() const noexcept { return source_; }
  /** @brief The schema it declares, `MODEL_SCHEMA`. */
  [[nodiscard]] const std::string& Schema() const noexcept { return schema_; }
  /** @brief The model's name, where it has one. */
  [[nodiscard]] const std::optional<std::string>& Name() const noexcept { return name_; }
  /** @brief The scales the numbers are in. */
  [[nodiscard]] const ::planetmodel::Scales& GetScales() const noexcept { return scales_; }
  /** @brief The geometry's tolerance, relative to the skeleton's span. */
  [[nodiscard]] double Rtol() const noexcept { return rtol_; }
  /** @brief The model's constants by name, in its units. */
  [[nodiscard]] const std::map<std::string, FileConstant>& Constants() const noexcept {
    return constants_;
  }
  /** @brief The writer's `meta`, by key. */
  [[nodiscard]] const netcdf::Attributes& Meta() const noexcept { return meta_; }
  /** @brief The reference skeleton. */
  [[nodiscard]] const ::planetmodel::Skeleton& GetSkeleton() const noexcept { return *skeleton_; }
  /** @brief Each layer's name, empty where it has none. */
  [[nodiscard]] const std::vector<std::optional<std::string>>& LayerNames() const noexcept {
    return layer_names_;
  }
  /** @brief Each interface's name, empty where it has none. */
  [[nodiscard]] const std::vector<std::optional<std::string>>& InterfaceNames() const noexcept {
    return interface_names_;
  }
  /** @brief Each interface's radius, centre outward. */
  [[nodiscard]] const std::vector<double>& InterfaceRadii() const noexcept {
    return interface_radii_;
  }
  /** @brief The parts, centre outward. */
  [[nodiscard]] const std::vector<FilePart>& Parts() const noexcept { return parts_; }
  /** @brief The `geometry` group, where the file holds the mapping so. */
  [[nodiscard]] const std::optional<FileGeometry>& Geometry() const noexcept { return geometry_; }

  /**
   * @brief The part holding a layer.
   * @param layer The layer.
   * @throws planetmodel::Error if no part holds it.
   */
  [[nodiscard]] const FilePart& PartOf(int layer) const {
    for (const auto& p : parts_) {
      if (std::ranges::contains(p.layers, layer)) return p;
    }
    throw Error(source_, "", std::format("no part holds layer {}", layer));
  }

  /**
   * @brief The names of the fields a layer holds.
   * @param layer The layer.
   * @throws planetmodel::Error if no part holds the layer.
   */
  [[nodiscard]] std::vector<std::string> Names(int layer) const {
    const auto& part = PartOf(layer);
    auto out = std::vector<std::string>{};
    for (const auto& [name, var] : part.variables) {
      if (IsFieldName(name) && Holds(part, layer, var)) out.push_back(name);
    }
    return out;
  }

  /**
   * @brief Whether a layer holds a field.
   * @param layer The layer.
   * @param name The field.
   */
  [[nodiscard]] bool HasField(int layer, std::string_view name) const {
    return std::ranges::contains(Names(layer), std::string{name});
  }

  /**
   * @brief A field of one variable on one layer, by the rules the file
   * states.
   * @param layer The layer.
   * @param name The variable.
   * @throws planetmodel::Error if the layer does not hold the variable as
   *         a field of the model, or its stated lateral rule is `harmonic`,
   *         which the C++ does not read.
   */
  [[nodiscard]] ::planetmodel::Field FieldOn(int layer, std::string_view name) const {
    const auto& part = PartOf(layer);
    const auto it = part.variables.find(std::string{name});
    if (name == "displacement") {
      throw Error(source_, std::format("part_{}/displacement", part.index),
                  "is the mapping, not a field of the model: read it with Mapping()");
    }
    if (it == part.variables.end() || !IsFieldName(it->first) ||
        !Holds(part, layer, it->second)) {
      throw Error(source_, std::format("part_{}/{}", part.index, name),
                  std::format("is not held on layer {}", layer));
    }
    const auto& var = it->second;
    auto out = ::planetmodel::Field{};
    out.source_ = source_;
    out.name_ = std::string{name};
    out.layer_ = layer;
    out.interval_ = skeleton_->Interval(static_cast<std::size_t>(layer));
    out.character_ = CharacterOf(var);
    out.dimensions_ = detail::DimensionsFrom(var.Numbers("dimensions"));
    out.reference_ = std::make_shared<const ::planetmodel::Field::Piece>(
        Piece(part, layer, var, out.character_));
    if (const auto other = var.Text("perturbation")) {
      const auto& pert = part.variables.at(*other);
      if (out.character_.Rank() != 0 || CharacterOf(pert).Rank() != 0) {
        throw Error(source_, std::format("part_{}/{}", part.index, name),
                    "is perturbed, and a perturbed field and its perturbation are of "
                    "rank 0");
      }
      if (Holds(part, layer, pert)) {
        out.kind_ = *var.Text("perturbation_kind");
        out.perturbation_ = std::make_shared<const ::planetmodel::Field::Piece>(
            Piece(part, layer, pert, CharacterOf(pert)));
      }
    }
    return out;
  }

  /**
   * @brief A field per layer that holds a variable, centre outward.
   * @param name The variable.
   * @throws planetmodel::Error as `FieldOn` does.
   */
  [[nodiscard]] std::vector<::planetmodel::Field> FieldsOf(std::string_view name) const {
    auto out = std::vector<::planetmodel::Field>{};
    for (auto layer = 0; layer < static_cast<int>(skeleton_->NLayers()); ++layer) {
      if (HasField(layer, name)) out.push_back(FieldOn(layer, name));
    }
    return out;
  }

  /**
   * @brief The mapping the file holds, the identity where it holds none.
   * @throws planetmodel::Error if a displacement field's lateral rule is
   *         `harmonic`.
   */
  [[nodiscard]] ModelMapping Mapping() const {
    auto out = ModelMapping{};
    out.source_ = source_;
    out.inner_ = skeleton_->Boundaries().front();
    out.outer_ = skeleton_->Boundaries().back();
    out.tolerance_ = rtol_ * skeleton_->Span();
    out.boundaries_ = skeleton_->Boundaries();
    if (geometry_) {
      out.form_ = ModelMapping::Form::Stack;
      out.radii_ = geometry_->radii;
      const auto& g = *geometry_->grid;
      const auto rule = LateralRuleNamed(geometry_->lateral, source_, "geometry");
      const auto cells = g.NTheta() * g.NPhi();
      for (auto k = std::size_t{0}; k < geometry_->radii.size(); ++k) {
        const auto values = std::span<const double>(geometry_->values).subspan(k * cells, cells);
        const auto poles = geometry_->poles
                               ? std::span<const double>(*geometry_->poles).subspan(2 * k, 2)
                               : std::span<const double>{};
        const auto zero = std::ranges::all_of(values, [](double v) { return v == 0.0; }) &&
                          std::ranges::all_of(poles, [](double v) { return v == 0.0; });
        if (zero) {
          out.reliefs_.emplace_back(std::nullopt);
        } else {
          out.reliefs_.emplace_back(LateralInterpolant(g, values, poles, rule, "geometry"));
        }
      }
      return out;
    }
    if (!parts_.empty() && parts_.front().variables.contains("displacement")) {
      out.form_ = ModelMapping::Form::Field;
      for (auto layer = 0; layer < static_cast<int>(skeleton_->NLayers()); ++layer) {
        out.fields_.push_back(Displacement(layer));
      }
    }
    return out;
  }

 private:
  friend ModelFile ReadModelFile(const std::filesystem::path& path);

  ModelFile() = default;

  /// Whether a variable is a field of the model, not a field's poles, a
  /// field's perturbation or the mapping.
  static bool IsFieldName(std::string_view name) {
    return !name.ends_with("_poles") && !name.ends_with("_perturbation") &&
           name != "displacement";
  }

  /// Whether a variable has a value on a layer: a field's fill is NaN.
  static bool Holds(const FilePart& part, int layer, const FileVariable& var) {
    const auto [e0, e1] = part.Elements(layer);
    const auto per_node = var.values.size() / var.shape.front();
    const auto begin = e0 * part.ngll * per_node;
    const auto end = e1 * part.ngll * per_node;
    return std::any_of(var.values.begin() + static_cast<std::ptrdiff_t>(begin),
                       var.values.begin() + static_cast<std::ptrdiff_t>(end),
                       [](double v) { return !std::isnan(v); });
  }

  /// The character a variable states; a displacement is a vector.
  static Character CharacterOf(const FileVariable& var) {
    if (var.name == "displacement") return Character(1, 0, false);
    auto integer = [&var](std::string_view key) {
      const auto v = var.Numbers(key);
      if (!v || v->size() != 1) throw Error("", var.name, std::format("{} is not a number", key));
      return static_cast<int>((*v)[0]);
    };
    return Character(integer("rank"), integer("weight"), integer("voigt") != 0);
  }


  /// The displacement field of a layer, a vector of the stated rules.
  ::planetmodel::Field Displacement(int layer) const {
    const auto& part = PartOf(layer);
    const auto& var = part.variables.at("displacement");
    auto out = ::planetmodel::Field{};
    out.source_ = source_;
    out.name_ = "displacement";
    out.layer_ = layer;
    out.interval_ = skeleton_->Interval(static_cast<std::size_t>(layer));
    out.character_ = Character(1, 0, false);
    out.reference_ = std::make_shared<const ::planetmodel::Field::Piece>(
        Piece(part, layer, var, out.character_));
    return out;
  }

  /// One variable on one layer as a piece of a field.
  ::planetmodel::Field::Piece Piece(const FilePart& part, int layer, const FileVariable& var,
                                    const Character& character) const {
    const auto where = std::format("part_{}/{}", part.index, var.name);
    auto out = ::planetmodel::Field::Piece{};
    out.character = character;
    out.rule = RadialRuleNamed(*var.Text("radial_interpolation"), source_, where);
    const auto [e0, e1] = part.Elements(layer);
    const auto ngll = part.ngll;
    const auto n0 = e0 * ngll;
    const auto n1 = e1 * ngll;
    const auto ncomp = character.StoredSize();
    auto radii = std::vector<double>(part.radius.begin() + static_cast<std::ptrdiff_t>(n0),
                                     part.radius.begin() + static_cast<std::ptrdiff_t>(n1));
    out.layout = RadialLayout(std::move(radii), ngll, rtol_ * skeleton_->Span());
    auto radial_layers = std::vector<int>{};
    if (const auto listed = var.Numbers("radial_layers")) {
      for (const auto x : *listed) radial_layers.push_back(static_cast<int>(x));
    }
    const auto lateral = var.IsLateral();
    out.radial = !lateral || std::ranges::contains(radial_layers, layer);
    if (out.radial) {
      // the stored components at every node, a lateral variable's from its
      // first column, all of whose columns are the same
      const auto stride = lateral ? part.grid->NTheta() * part.grid->NPhi() * ncomp : ncomp;
      for (auto node = n0; node < n1; ++node) {
        const auto* at = &var.values[node * stride];
        out.nodal.insert(out.nodal.end(), at, at + ncomp);
      }
      return out;
    }
    const auto& grid = *part.grid;
    const auto rule = LateralRuleNamed(*var.Text("lateral_interpolation"), source_, where);
    const auto nt = grid.NTheta();
    const auto np = grid.NPhi();
    const auto* poles = part.variables.contains(var.name + "_poles")
                            ? &part.variables.at(var.name + "_poles")
                            : nullptr;
    const auto rank = character.Rank();
    // the frames at the grid's nodes, for components of rank one and above
    auto frames = std::vector<std::vector<Matrix3>>{};
    if (rank > 0) {
      for (auto i = std::size_t{0}; i < nt; ++i) {
        for (auto j = std::size_t{0}; j < np; ++j) {
          frames.emplace_back(static_cast<std::size_t>(rank),
                              SphericalFrame(grid.Colatitudes()[i], grid.Longitudes()[j]));
        }
      }
    }
    for (auto node = n0; node < n1; ++node) {
      // the node's values, Cartesian, laid out by component
      auto cart = std::vector<double>(ncomp * nt * np);
      for (auto cell = std::size_t{0}; cell < nt * np; ++cell) {
        const auto at = std::span<const double>(var.values).subspan((node * nt * np + cell) * ncomp,
                                                                     ncomp);
        const auto turned = rank > 0 ? RotateStored(at, rank, character.Voigt(), frames[cell])
                                     : std::vector<double>(at.begin(), at.end());
        for (auto c = std::size_t{0}; c < ncomp; ++c) cart[c * nt * np + cell] = turned[c];
      }
      for (auto c = std::size_t{0}; c < ncomp; ++c) {
        auto at_poles = std::vector<double>{};
        if (poles) {
          for (auto k = 0; k < 2; ++k) {
            at_poles.push_back(poles->values[(node * 2 + static_cast<std::size_t>(k)) * ncomp + c]);
          }
        }
        out.lateral.emplace_back(grid, std::span<const double>(cart).subspan(c * nt * np, nt * np),
                                 at_poles, rule, where);
      }
    }
    return out;
  }

  std::string source_{};
  std::string schema_{};
  std::optional<std::string> name_{};
  ::planetmodel::Scales scales_{};
  double rtol_ = DEFAULT_RTOL;
  std::map<std::string, FileConstant> constants_{};
  netcdf::Attributes meta_{};
  std::optional<::planetmodel::Skeleton> skeleton_{};
  std::vector<std::optional<std::string>> layer_names_{};
  std::vector<std::optional<std::string>> interface_names_{};
  std::vector<double> interface_radii_{};
  std::vector<FilePart> parts_{};
  std::optional<FileGeometry> geometry_{};
};

namespace detail {

/// The angular grid a group's coordinates and attributes hold, as the
/// Python's `_grid_of` reads it: colatitude 90 - latitude in radians,
/// clipped to [0, pi]; longitude in radians reduced into [0, 2 pi).
inline AngularGrid GridOf(int group, const netcdf::Attributes& attrs, const std::string& source) {
  constexpr auto degree = std::numbers::pi / 180.0;
  auto theta = netcdf::ReadDoubles(group, "latitude", source).values;
  for (auto& t : theta) t = std::clamp((90.0 - t) * degree, 0.0, std::numbers::pi);
  auto phi = netcdf::ReadDoubles(group, "longitude", source).values;
  for (auto& p : phi) p = Wrapped(p * degree);
  auto lmax = std::optional<int>{};
  if (attrs.contains("grid_lmax")) {
    const auto l = NumberOf(attrs, "grid_lmax");
    if (!l) Fail(source, "grid_lmax", "is not a degree");
    lmax = static_cast<int>(*l);
  }
  return AngularGrid(std::move(theta), std::move(phi),
                     TextOf(attrs, "grid_kind").value_or("custom"), lmax);
}

inline FileVariable ReadVariable(int group, const std::string& name, const std::string& source) {
  auto out = FileVariable{.name = name};
  auto array = netcdf::ReadDoubles(group, name, source);
  out.values = std::move(array.values);
  out.shape = std::move(array.shape);
  out.dimensions = netcdf::DimensionNames(group, name, source);
  out.attributes = netcdf::ReadAttributes(group, netcdf::Variable(group, name, source), source);
  return out;
}

}  // namespace detail

/**
 * @brief A model file, read whole and checked as the Python's
 * `validate_structure` checks it.
 * @param path The file.
 * @return Its numbers and metadata, as stored.
 * @throws planetmodel::Error naming the file and the place of the first
 *         fault, or netCDF-C's message where the file cannot be read.
 */
[[nodiscard]] inline ModelFile ReadModelFile(const std::filesystem::path& path) {
  const auto file = netcdf::File(path);
  const auto root = file.Id();
  const auto& source = file.Path();
  auto fail = [&source](std::string_view where, std::string_view message) {
    detail::Fail(source, where, message);
  };
  auto out = ModelFile{};
  out.source_ = source;
  const auto attrs = netcdf::ReadAttributes(root, netcdf::GLOBAL, source);
  auto text = [&attrs](std::string_view key) { return detail::TextOf(attrs, key); };
  auto number = [&attrs](std::string_view key) { return detail::NumberOf(attrs, key); };

  // the root
  const auto schema = text("schema");
  if (!schema) fail("schema", "is missing: not a planetmodel model file");
  if (*schema != MODEL_SCHEMA) {
    fail("schema", std::format("is {}, and this reader reads {}", *schema, MODEL_SCHEMA));
  }
  out.schema_ = *schema;
  out.name_ = text("name");
  auto scales = std::array<double, 4>{};
  constexpr auto bases = std::array<std::string_view, 4>{"length", "mass", "time", "temperature"};
  for (auto b = std::size_t{0}; b < 4; ++b) {
    const auto key = std::format("scale_{}", bases[b]);
    const auto value = number(key);
    if (!value || !(*value > 0.0)) fail(key, "is missing or not positive");
    scales[b] = *value;
  }
  out.scales_ = Scales({.length = scales[0], .mass = scales[1], .time = scales[2],
                        .temperature = scales[3]});
  if (attrs.contains("rtol")) {
    const auto rtol = number("rtol");
    if (!rtol || !(*rtol > 0.0 && *rtol < 1.0)) fail("rtol", "is not a tolerance in (0, 1)");
    out.rtol_ = *rtol;
  }
  for (const auto& [key, value] : attrs) {
    if (key.starts_with("meta_")) out.meta_.emplace(key.substr(5), value);
    if (!key.starts_with("constant_") || key.ends_with("_dimensions") ||
        key.ends_with("_meaning")) {
      continue;
    }
    const auto cname = key.substr(9);
    const auto v = number(key);
    if (!v) fail(key, "is not a number");
    auto constant = FileConstant{.value = *v};
    if (attrs.contains(key + "_dimensions")) {
      constant.dimensions = detail::DimensionsFrom(detail::NumbersOf(attrs, key + "_dimensions"));
      if (!constant.dimensions) fail(key + "_dimensions", "is not four exponents");
    }
    constant.meaning = text(key + "_meaning").value_or("");
    out.constants_.emplace(cname, constant);
  }

  // the skeleton
  for (const auto& [name, dims] : std::vector<std::pair<std::string, std::vector<std::string>>>{
           {"layer_bounds", {"layer", "bound"}},
           {"layer_name", {"layer"}},
           {"interface_radius", {"interface"}},
           {"interface_name", {"interface"}},
           {"layer_part", {"layer"}}}) {
    if (!netcdf::HasVariable(root, name)) fail(name, "is not a variable at the root");
    if (netcdf::DimensionNames(root, name, source) != dims) {
      fail(name, "has the wrong dimensions");
    }
  }
  const auto bounds = netcdf::ReadDoubles(root, "layer_bounds", source).values;
  const auto nlayers = bounds.size() / 2;
  auto boundaries = std::vector<double>{};
  for (auto i = std::size_t{0}; i < nlayers; ++i) {
    if (!(bounds[2 * i + 1] > bounds[2 * i]) || (i > 0 && bounds[2 * i] != bounds[2 * i - 1])) {
      fail("layer_bounds", "do not make a skeleton of contiguous layers");
    }
    boundaries.push_back(bounds[2 * i]);
  }
  boundaries.push_back(bounds.back());
  out.skeleton_ = Skeleton(boundaries);
  for (const auto& n : netcdf::ReadStrings(root, "layer_name", source)) {
    out.layer_names_.push_back(n.empty() ? std::nullopt : std::optional<std::string>{n});
  }
  for (const auto& n : netcdf::ReadStrings(root, "interface_name", source)) {
    out.interface_names_.push_back(n.empty() ? std::nullopt : std::optional<std::string>{n});
  }
  out.interface_radii_ = netcdf::ReadDoubles(root, "interface_radius", source).values;
  const auto layer_part_values = netcdf::ReadDoubles(root, "layer_part", source).values;
  auto layer_part = std::vector<int>{};
  for (const auto x : layer_part_values) layer_part.push_back(static_cast<int>(x));
  const auto groups = netcdf::GroupNames(root, source);
  auto part_numbers = layer_part;
  std::ranges::sort(part_numbers);
  part_numbers.erase(std::unique(part_numbers.begin(), part_numbers.end()), part_numbers.end());
  for (const auto k : part_numbers) {
    const auto gname = std::format("part_{}", k);
    if (!std::ranges::contains(groups, gname)) {
      fail("layer_part", std::format("names {}, which the file lacks", gname));
    }
    auto layers = std::vector<int>{};
    for (auto i = std::size_t{0}; i < nlayers; ++i) {
      if (layer_part[i] == k) layers.push_back(static_cast<int>(i));
    }
    if (layers.back() - layers.front() + 1 != static_cast<int>(layers.size())) {
      fail("layer_part", std::format("the layers of part {} are not contiguous", k));
    }
    out.parts_.push_back(FilePart{.index = k, .layers = layers});
  }

  // the parts
  for (auto& part : out.parts_) {
    const auto where = std::format("part_{}", part.index);
    const auto g = netcdf::Group(root, where, source);
    for (const auto* dim : {"node", "element", "bound", "latitude", "longitude"}) {
      if (!netcdf::DimensionLength(g, dim, source)) {
        fail(where, std::format("has no dimension {}", dim));
      }
    }
    for (const auto* name :
         {"radius", "element_layer", "element_bounds", "latitude", "longitude"}) {
      if (!netcdf::HasVariable(g, name)) fail(where, std::format("has no variable {}", name));
    }
    const auto gattrs = netcdf::ReadAttributes(g, netcdf::GLOBAL, source);
    if (!gattrs.contains("ngll") || !gattrs.contains("grid_kind")) {
      fail(where, "lacks the attribute ngll or grid_kind");
    }
    const auto kind = detail::TextOf(gattrs, "grid_kind");
    if (!kind || !std::ranges::contains(GRID_KINDS, std::string_view{*kind})) {
      fail(where + ".grid_kind",
           std::format("is {}, not a kind of grid", kind.value_or("a number")));
    }
    const auto ngll = detail::NumberOf(gattrs, "ngll");
    if (!ngll || !(*ngll >= 2.0)) fail(where + ".ngll", "is not a number of nodes, two or more");
    part.ngll = static_cast<std::size_t>(*ngll);
    const auto nelem = *netcdf::DimensionLength(g, "element", source);
    const auto nnode = *netcdf::DimensionLength(g, "node", source);
    if (part.ngll < 2 || nelem * part.ngll != nnode) {
      fail(where, std::format("{} nodes are not {} elements of ngll = {}", nnode, nelem,
                              part.ngll));
    }
    for (const auto x : netcdf::ReadDoubles(g, "element_layer", source).values) {
      part.element_layer.push_back(static_cast<int>(x));
    }
    auto held = part.element_layer;
    std::ranges::sort(held);
    held.erase(std::unique(held.begin(), held.end()), held.end());
    if (held != part.layers) fail(where, "its elements do not lie in its layers");
    part.element_bounds = netcdf::ReadDoubles(g, "element_bounds", source).values;
    const auto& eb = part.element_bounds;
    // the part's ends and every element's ends meet to 1e-12 of its span
    const auto bottom = bounds[2 * static_cast<std::size_t>(part.layers.front())];
    const auto top = bounds[2 * static_cast<std::size_t>(part.layers.back()) + 1];
    const auto tile = 1e-12 * (top - bottom);
    auto tiles = std::abs(eb.front() - bottom) <= tile && std::abs(eb.back() - top) <= tile;
    for (auto e = std::size_t{1}; e < nelem; ++e) {
      tiles = tiles && std::abs(eb[2 * e] - eb[2 * e - 1]) <= tile;
    }
    if (!tiles) fail(where, "element_bounds do not tile its layers");
    part.radius = netcdf::ReadDoubles(g, "radius", source).values;
    for (auto e = std::size_t{0}; e < nelem; ++e) {
      const auto nodes = GllNodes(eb[2 * e], eb[2 * e + 1], part.ngll);
      for (auto k = std::size_t{0}; k < part.ngll; ++k) {
        const auto stored = part.radius[e * part.ngll + k];
        if (std::abs(stored - nodes[k]) > 1e-12 * std::abs(nodes[k])) {
          fail(where + "/radius", "is not the Gauss-Lobatto-Legendre nodes of its elements");
        }
      }
    }
    part.grid = detail::GridOf(g, gattrs, source);
    if (netcdf::HasVariable(g, "colatitude_weights")) {
      part.colatitude_weights = netcdf::ReadDoubles(g, "colatitude_weights", source).values;
    }
    const auto lacks_pole = netcdf::DimensionLength(g, "pole", source).has_value();
    for (const auto& vname : netcdf::VariableNames(g, source)) {
      if (vname == "radius" || vname == "element_layer" || vname == "element_bounds" ||
          vname == "latitude" || vname == "longitude" || vname == "colatitude_weights") {
        continue;
      }
      auto var = detail::ReadVariable(g, vname, source);
      const auto vwhere = std::format("{}/{}", where, vname);
      if (vname.ends_with("_poles")) {
        part.variables.emplace(vname, std::move(var));
        continue;
      }
      if (var.dimensions.empty() || var.dimensions.front() != "node") {
        fail(vwhere, "does not run over the nodes");
      }
      const auto lateral = var.IsLateral();
      const auto need = vname == "displacement"
                            ? std::vector<std::string>{"radial_interpolation"}
                            : std::vector<std::string>{"radial_interpolation", "rank", "weight",
                                                       "voigt"};
      for (const auto& key : need) {
        if (!var.attributes.contains(key)) fail(vwhere, std::format("lacks the attribute {}", key));
      }
      auto stored = std::vector<std::size_t>{};
      try {
        stored = vname == "displacement" ? std::vector<std::size_t>{3}
                                         : ModelFile::CharacterOf(var).StoredShape();
      } catch (const Error& e) {
        fail(vwhere, std::format("states no character: {}", e.what()));
      }
      const auto tail = std::vector<std::size_t>(var.shape.begin() + (lateral ? 3 : 1),
                                                 var.shape.end());
      if (tail != stored) fail(vwhere, "has trailing axes its character does not store");
      const auto radial_rule = var.Text("radial_interpolation");
      if (!radial_rule || !std::ranges::contains(RADIAL_RULES, std::string_view{*radial_rule})) {
        fail(vwhere, "states an unknown radial rule");
      }
      if (var.attributes.contains("radial_layers")) {
        auto holders = std::vector<int>{};
        if (const auto l = var.Numbers("layers")) {
          for (const auto x : *l) holders.push_back(static_cast<int>(x));
        } else {
          holders = part.layers;
        }
        for (const auto x : var.Numbers("radial_layers").value_or(std::vector<double>{-1.0})) {
          if (!lateral || !std::ranges::contains(holders, static_cast<int>(x))) {
            fail(vwhere, "names radial_layers which must be layers a lateral variable holds");
          }
        }
      }
      if (lateral) {
        const auto rule = var.Text("lateral_interpolation");
        if (!rule || !std::ranges::contains(LATERAL_RULES, std::string_view{*rule})) {
          fail(vwhere, "states no known lateral rule");
        }
        if (lacks_pole && !netcdf::HasVariable(g, vname + "_poles")) {
          fail(vwhere, std::format("varies with direction on a grid without its poles, and "
                                   "{}_poles is missing",
                                   vname));
        }
      }
      if (var.attributes.contains("perturbation")) {
        const auto other = var.Text("perturbation");
        if (!other || !netcdf::HasVariable(g, *other)) {
          fail(vwhere, std::format("names its perturbation {}, which the part lacks",
                                   other.value_or("by a number")));
        }
        const auto kind = var.Text("perturbation_kind");
        if (!kind || !std::ranges::contains(PERTURBATION_KINDS, std::string_view{*kind})) {
          fail(vwhere, "states no known perturbation_kind");
        }
      }
      part.variables.emplace(vname, std::move(var));
    }
  }

  // the mapping
  auto holding = 0;
  for (const auto& part : out.parts_) holding += part.variables.contains("displacement");
  const auto has_stack = std::ranges::contains(groups, std::string{"geometry"});
  if (holding != 0 && holding != static_cast<int>(out.parts_.size())) {
    fail("displacement", "is held in some parts only; a mapping held as a field is held in "
                         "every part");
  }
  if (has_stack && holding != 0) {
    fail("geometry", "the mapping is held twice, as the geometry group and as a displacement "
                     "field");
  }
  if (has_stack) {
    const auto g = netcdf::Group(root, "geometry", source);
    for (const auto* name : {"boundary_radius", "boundary_displacement", "latitude", "longitude"}) {
      if (!netcdf::HasVariable(g, name)) fail("geometry", std::format("has no variable {}", name));
    }
    if (netcdf::DimensionNames(g, "boundary_displacement", source) !=
        std::vector<std::string>{"boundary", "latitude", "longitude"}) {
      fail("geometry/boundary_displacement", "has the wrong dimensions");
    }
    const auto var = detail::ReadVariable(g, "boundary_displacement", source);
    const auto lateral = var.Text("lateral_interpolation");
    if (!lateral || !std::ranges::contains(LATERAL_RULES, std::string_view{*lateral})) {
      fail("geometry/boundary_displacement", "states no known lateral rule");
    }
    if (var.Text("between") != "linear") {
      fail("geometry/boundary_displacement", "states an unknown rule between surfaces");
    }
    auto geometry = FileGeometry{};
    geometry.radii = netcdf::ReadDoubles(g, "boundary_radius", source).values;
    geometry.grid = detail::GridOf(g, netcdf::ReadAttributes(g, netcdf::GLOBAL, source), source);
    geometry.values = var.values;
    if (netcdf::HasVariable(g, "boundary_displacement_poles")) {
      geometry.poles = netcdf::ReadDoubles(g, "boundary_displacement_poles", source).values;
    }
    geometry.lateral = *lateral;
    geometry.between = "linear";
    out.geometry_ = std::move(geometry);
  }
  return out;
}

}  // namespace planetmodel

#endif  // PLANETMODEL_MODELFILE_HPP
