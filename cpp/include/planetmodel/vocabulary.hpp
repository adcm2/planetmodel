#ifndef PLANETMODEL_VOCABULARY_HPP
#define PLANETMODEL_VOCABULARY_HPP

/**
 * @file
 * @brief The shipped field names, as `planetmodel.vocabulary` defines
 * them.
 *
 * A name of the vocabulary means one thing in every model: its character,
 * its dimensions and a gloss.  The Python's `read_model` refuses a file
 * that states a vocabulary name otherwise, so the writer takes a name's
 * spec from here and refuses another.  Restated from the Python, which a
 * fixture holds it to.
 */

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include "planetmodel/character.hpp"
#include "planetmodel/units.hpp"

namespace planetmodel {

/** @brief What a field name means: its character, dimensions and a gloss. */
struct FieldSpec {
  Character character = SCALAR;           /**< Its law under a mapping. */
  std::optional<Dimensions> dimensions{}; /**< Its dimensions, where they are declared. */
  std::string meaning{};                  /**< What it is. */
};

/** @brief The shipped names, each with its spec. */
[[nodiscard]] inline const std::map<std::string, FieldSpec, std::less<>>& Vocabulary() {
  namespace u = units;
  static const auto vocabulary = std::map<std::string, FieldSpec, std::less<>>{
      {"rho", {DENSITY, u::DENSITY, "mass density"}},
      {"vp", {SCALAR, u::VELOCITY, "P-wave speed of an isotropic medium"}},
      {"vs", {SCALAR, u::VELOCITY, "S-wave speed of an isotropic medium"}},
      {"vpv", {SCALAR, u::VELOCITY, "P-wave speed along the symmetry axis (vertical)"}},
      {"vsv",
       {SCALAR, u::VELOCITY, "S-wave speed polarised along the symmetry axis (vertical)"}},
      {"vph", {SCALAR, u::VELOCITY, "P-wave speed normal to the symmetry axis (horizontal)"}},
      {"vsh",
       {SCALAR, u::VELOCITY,
        "S-wave speed polarised normal to the symmetry axis (horizontal)"}},
      {"eta", {SCALAR, u::DIMENSIONLESS, "the anisotropy parameter F / (A - 2L)"}},
      {"qkappa", {SCALAR, u::DIMENSIONLESS, "bulk quality factor; zero attenuates nothing"}},
      {"qmu",
       {SCALAR, u::DIMENSIONLESS, "shear quality factor; zero attenuates nothing (a fluid)"}},
      {"qp", {SCALAR, u::DIMENSIONLESS, "quality factor of P waves; zero attenuates nothing"}},
      {"kappa", {DENSITY, u::MODULUS, "bulk modulus of an isotropic medium"}},
      {"mu", {DENSITY, u::MODULUS, "shear modulus of an isotropic medium"}},
      {"A", {DENSITY, u::MODULUS, "Love modulus A = rho vph^2"}},
      {"C", {DENSITY, u::MODULUS, "Love modulus C = rho vpv^2"}},
      {"F", {DENSITY, u::MODULUS, "Love modulus F = eta (A - 2L)"}},
      {"L", {DENSITY, u::MODULUS, "Love modulus L = rho vsv^2"}},
      {"N", {DENSITY, u::MODULUS, "Love modulus N = rho vsh^2"}},
      {"elastic_moduli",
       {ELASTIC, u::MODULUS,
        "the second elasticity tensor, static, Voigt-reduced in the spherical frame"}},
      {"viscosity", {SCALAR, u::VISCOSITY, "dynamic viscosity"}},
      {"temperature", {SCALAR, u::TEMPERATURE, "absolute temperature"}},
      {"g",
       {SCALAR, u::GRAVITY,
        "gravitational acceleration of the spherically symmetric reference, radially "
        "inward, by magnitude"}},
      {"drho_dphi0",
       {DENSITY, Dimensions{.mass = 1, .length = -5, .time = 2},
        "stratification of a fluid layer: d rho / d Phi_0 = rho'(r) / g(r)"}},
  };
  return vocabulary;
}

/**
 * @brief The spec of a vocabulary name.
 * @param name The name.
 * @return Its spec, or empty where the vocabulary has no such name.
 */
[[nodiscard]] inline std::optional<FieldSpec> VocabularySpec(std::string_view name) {
  const auto& vocabulary = Vocabulary();
  const auto it = vocabulary.find(name);
  if (it == vocabulary.end()) return std::nullopt;
  return it->second;
}

}  // namespace planetmodel

#endif  // PLANETMODEL_VOCABULARY_HPP
