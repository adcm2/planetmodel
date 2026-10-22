#ifndef PLANETMODEL_RHEOLOGY_HPP
#define PLANETMODEL_RHEOLOGY_HPP

/**
 * @file
 * @brief Prepared elastic moduli with a constant-Q absorption band.
 */

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <optional>
#include <string_view>

#include "planetmodel/radial_model.hpp"

namespace planetmodel {

/** @brief Love moduli A, C, F, L and N. */
template <typename T>
struct ElasticModuli {
  T A{}; /**< Tangential or horizontal compression modulus. */
  T C{}; /**< Radial normal modulus. */
  T F{}; /**< Radial-tangential coupling modulus. */
  T L{}; /**< Radial shear modulus. */
  T N{}; /**< Tangential shear modulus. */
};

/** @brief Complex Love moduli in named A, C, F, L, N members. */
using ComplexModuli = ElasticModuli<std::complex<double>>;

/** @brief The constant-Q dispersion factor for exp(+i omega t). */
[[nodiscard]] inline std::complex<double> ConstantQFactor(
    std::complex<double> omega, double omega_ref) {
  if (!(std::isfinite(omega.real()) && std::isfinite(omega.imag()) &&
        (omega.real() != 0.0 || omega.imag() != 0.0) && omega.imag() <= 0.0))
    throw Error("ConstantQFactor", "omega", "must be finite, nonzero and in the lower half-plane");
  if (!(std::isfinite(omega_ref) && omega_ref > 0.0))
    throw Error("ConstantQFactor", "omega_ref", "must be positive and finite");
  // log(i omega / omega_ref), arranged to avoid overflow in the ratio.
  const std::complex<double> rotated{-omega.imag(), omega.real()};
  return (2.0 / std::numbers::pi) * (std::log(rotated) - std::log(omega_ref));
}

/** @brief Elastic coefficients prepared for repeated constant-Q frequency evaluation. */
class PreparedConstantQ {
 public:
  /** @brief Cache the five elastic moduli and the Voigt loss coefficients at radius r. */
  PreparedConstantQ(const RadialMaterial& material, double r,
                   std::optional<double> omega_ref = {}) {
    for (const auto& [name, field] : material.Fields()) {
      static_cast<void>(field);
      if (Unsupported(name))
        throw Error("PreparedConstantQ", name, "only elastic and constant-Q fields are supported");
    }
    elastic_ = {material.Evaluate("A", r).value, material.Evaluate("C", r).value,
                material.Evaluate("F", r).value, material.Evaluate("L", r).value,
                material.Evaluate("N", r).value};
    for (const auto value : {elastic_.A, elastic_.C, elastic_.F, elastic_.L, elastic_.N}) {
      if (!std::isfinite(value)) throw Error("PreparedConstantQ", "moduli", "must be finite");
    }
    const auto qkappa = Quality(material, "qkappa", r);
    const auto qmu = Quality(material, "qmu", r);
    const auto kappa = material.Evaluate("kappa", r).value;
    const auto mu = material.Evaluate("mu", r).value;
    bulk_loss_ = qkappa > 0.0 ? kappa / qkappa : 0.0;
    shear_loss_ = qmu > 0.0 ? mu / qmu : 0.0;
    if (!(std::isfinite(bulk_loss_) && std::isfinite(shear_loss_)))
      throw Error("PreparedConstantQ", "loss coefficients", "must be finite");
    attenuated_ = bulk_loss_ != 0.0 || shear_loss_ != 0.0;
    if (attenuated_ && !omega_ref)
      throw Error("PreparedConstantQ", "omega_ref", "attenuation needs a reference angular frequency");
    if (omega_ref && !(std::isfinite(*omega_ref) && *omega_ref > 0.0))
      throw Error("PreparedConstantQ", "omega_ref", "must be positive and finite");
    omega_ref_ = omega_ref;
  }

  /** @brief Whether either modulus has a nonzero constant-Q loss. */
  [[nodiscard]] bool HasAttenuation() const noexcept {
    return attenuated_;
  }
  /** @brief The real, frequency-independent elastic moduli in A,C,F,L,N order. */
  [[nodiscard]] const ElasticModuli<double>& Elastic() const noexcept { return elastic_; }
  /** @brief Coefficient multiplying the shared frequency factor for bulk loss. */
  [[nodiscard]] double BulkLossCoefficient() const noexcept { return bulk_loss_; }
  /** @brief Coefficient multiplying the shared frequency factor for shear loss. */
  [[nodiscard]] double ShearLossCoefficient() const noexcept { return shear_loss_; }

  /** @brief Apply a factor shared by all radii to these cached coefficients. */
  [[nodiscard]] ComplexModuli ApplyFactor(std::complex<double> factor) const {
    if (!(std::isfinite(factor.real()) && std::isfinite(factor.imag())))
      throw Error("PreparedConstantQ", "factor", "must be finite");
    const auto dk = bulk_loss_ * factor;
    const auto dm = shear_loss_ * factor;
    return {elastic_.A + dk + (4.0 / 3.0) * dm,
            elastic_.C + dk + (4.0 / 3.0) * dm,
            elastic_.F + dk - (2.0 / 3.0) * dm,
            elastic_.L + dm, elastic_.N + dm};
  }

  /** @brief Evaluate at one complex angular frequency. */
  [[nodiscard]] ComplexModuli Evaluate(std::complex<double> omega) const {
    if (!HasAttenuation()) {
      if (!(std::isfinite(omega.real()) && std::isfinite(omega.imag()) &&
            (omega.real() != 0.0 || omega.imag() != 0.0) && omega.imag() <= 0.0))
        throw Error("PreparedConstantQ", "omega", "must be finite, nonzero and in the lower half-plane");
      return ApplyFactor({});
    }
    return ApplyFactor(ConstantQFactor(omega, *omega_ref_));
  }

 private:
  [[nodiscard]] static bool Unsupported(std::string_view name) {
    return name == "viscosity" || name == "stress_exponent" || name == "transition_stress" ||
           name.starts_with("relaxation_strength_") || name.starts_with("relaxation_time_");
  }
  [[nodiscard]] static double Quality(const RadialMaterial& material,
                                      std::string_view name, double r) {
    if (!material.Has(name)) return 0.0;
    const auto q = material.Evaluate(name, r).value;
    if (!(std::isfinite(q) && q >= 0.0))
      throw Error("PreparedConstantQ", name, "must be finite and non-negative");
    return q;
  }

  ElasticModuli<double> elastic_{};
  double bulk_loss_ = 0.0;
  double shear_loss_ = 0.0;
  bool attenuated_ = false;
  std::optional<double> omega_ref_;
};

}  // namespace planetmodel

#endif  // PLANETMODEL_RHEOLOGY_HPP
