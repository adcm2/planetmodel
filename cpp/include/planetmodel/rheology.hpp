#ifndef PLANETMODEL_RHEOLOGY_HPP
#define PLANETMODEL_RHEOLOGY_HPP

/**
 * @file
 * @brief Prepared elastic moduli with a constant-Q absorption band.
 */

#include <array>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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

/** @brief How TI moduli respond to a bulk/shear attenuation increment. */
enum class TIPrescription { Additive, FixedRatios };

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
                   std::optional<double> omega_ref = {},
                   TIPrescription ti_prescription = TIPrescription::Additive)
      : ti_prescription_(ti_prescription) {
    if (ti_prescription_ != TIPrescription::Additive &&
        ti_prescription_ != TIPrescription::FixedRatios)
      throw Error("PreparedConstantQ", "ti_prescription",
                  "must be Additive or FixedRatios");
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
    if (ti_prescription_ == TIPrescription::FixedRatios) PrepareFixedRatios();
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
  /** @brief The selected TI attenuation prescription. */
  [[nodiscard]] TIPrescription Prescription() const noexcept {
    return ti_prescription_;
  }

  /** @brief Apply a factor shared by all radii to these cached coefficients. */
  [[nodiscard]] ComplexModuli ApplyFactor(std::complex<double> factor) const {
    if (!(std::isfinite(factor.real()) && std::isfinite(factor.imag())))
      throw Error("PreparedConstantQ", "factor", "must be finite");
    if (ti_prescription_ == TIPrescription::FixedRatios)
      return {elastic_.A + fixed_loss_.A * factor, elastic_.C + fixed_loss_.C * factor,
              elastic_.F + fixed_loss_.F * factor, elastic_.L + fixed_loss_.L * factor,
              elastic_.N + fixed_loss_.N * factor};
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
  void PrepareFixedRatios() {
    const auto scale = std::max({std::abs(elastic_.A), std::abs(elastic_.C),
                                 std::abs(elastic_.F), std::abs(elastic_.L),
                                 std::abs(elastic_.N)});
    const auto tol = 128.0 * std::numeric_limits<double>::epsilon() * scale;
    const bool fluid = elastic_.L == 0.0 && elastic_.N == 0.0;
    if (fluid) {
      if (std::abs(elastic_.A - elastic_.C) > tol ||
          std::abs(elastic_.A - elastic_.F) > tol)
        throw Error("PreparedConstantQ", "fixed_ratios",
                    "zero-shear material must be a true isotropic fluid");
      fixed_loss_ = {bulk_loss_, bulk_loss_, bulk_loss_, 0.0, 0.0};
      if (bulk_loss_ < 0.0)
        throw Error("PreparedConstantQ", "fixed_ratios", "fluid loss is not passive");
      return;
    }
    if (std::abs(elastic_.A) <= tol || std::abs(elastic_.L) <= tol ||
        std::abs(elastic_.A - 2.0 * elastic_.L) <= tol)
      throw Error("PreparedConstantQ", "fixed_ratios", "TI ratios are undefined");
    const auto phi = elastic_.C / elastic_.A;
    const auto xi = elastic_.N / elastic_.L;
    const auto eta = elastic_.F / (elastic_.A - 2.0 * elastic_.L);
    const auto a11 = (4.0 + phi + 4.0 * eta) / 9.0;
    const auto a12 = -(8.0 * eta + 4.0 * xi) / 9.0;
    const auto a21 = (1.0 + phi - 2.0 * eta) / 15.0;
    const auto a22 = (4.0 * eta + 5.0 * xi + 6.0) / 15.0;
    const auto det = a11 * a22 - a12 * a21;
    const auto det_scale = std::max(std::abs(a11 * a22) + std::abs(a12 * a21),
                                    std::numeric_limits<double>::min());
    if (!(std::isfinite(det) && std::abs(det) >
          128.0 * std::numeric_limits<double>::epsilon() * det_scale))
      throw Error("PreparedConstantQ", "fixed_ratios", "TI mapping is singular or ill-conditioned");
    const auto da = (bulk_loss_ * a22 - a12 * shear_loss_) / det;
    const auto dl = (a11 * shear_loss_ - a21 * bulk_loss_) / det;
    fixed_loss_ = {da, phi * da, eta * (da - 2.0 * dl), dl, xi * dl};
    const auto loss_scale = std::max({std::abs(fixed_loss_.A), std::abs(fixed_loss_.C),
                                      std::abs(fixed_loss_.F), std::abs(fixed_loss_.L),
                                      std::abs(fixed_loss_.N)});
    if (!(std::isfinite(fixed_loss_.A) && std::isfinite(fixed_loss_.C) &&
          std::isfinite(fixed_loss_.F) && std::isfinite(fixed_loss_.L) &&
          std::isfinite(fixed_loss_.N)))
      throw Error("PreparedConstantQ", "fixed_ratios", "loss coefficients must be finite");
    const auto norm = loss_scale > 0.0 ? loss_scale : 1.0;
    const auto aa = (fixed_loss_.A - fixed_loss_.N) / norm;
    const auto cc = fixed_loss_.C / norm;
    const auto ff = fixed_loss_.F / norm;
    const auto minor = 2.0 * aa * cc - 2.0 * ff * ff;
    const auto minor_scale = std::abs(2.0 * aa * cc) + 2.0 * ff * ff;
    const auto loss_tol = 128.0 * std::numeric_limits<double>::epsilon();
    const auto minor_tol = 128.0 * std::numeric_limits<double>::epsilon() *
                           std::max(minor_scale, std::numeric_limits<double>::min());
    if (fixed_loss_.L / norm < -loss_tol || fixed_loss_.N / norm < -loss_tol || aa < -loss_tol ||
        cc < -loss_tol || minor < -minor_tol)
      throw Error("PreparedConstantQ", "fixed_ratios", "TI loss tensor is not passive");
  }

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
  TIPrescription ti_prescription_ = TIPrescription::Additive;
  ElasticModuli<double> fixed_loss_{};
};

}  // namespace planetmodel

#endif  // PLANETMODEL_RHEOLOGY_HPP
