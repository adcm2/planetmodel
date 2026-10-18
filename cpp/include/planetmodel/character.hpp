#ifndef PLANETMODEL_CHARACTER_HPP
#define PLANETMODEL_CHARACTER_HPP

/// @file
/// A field's character, its law under a mapping, as `planetmodel.character`
/// defines it: the rank (the number of factors of F), the weight (the power
/// of 1/J, 0 or 1), and whether a rank-2 or rank-4 value is stored in Voigt
/// form.  The stored shape follows: nothing for a scalar, 3 for a vector,
/// 6 or 6 by 6 in Voigt form for ranks 2 and 4, 3 per index otherwise.

#include <cstddef>
#include <format>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel {

/// The transformation law of a field: rank, weight, and Voigt form.
class Character {
 public:
  /// A character of rank `rank` >= 0 and weight 0 or 1, refused otherwise.
  explicit constexpr Character(int rank, int weight = 0, bool voigt = true)
      : rank_(rank), weight_(weight), voigt_(voigt) {
    if (rank < 0) {
      throw Error("Character", "", std::format("rank must be non-negative, got {}", rank));
    }
    if (weight != 0 && weight != 1) {
      throw Error("Character", "", std::format("weight must be 0 or 1, got {}", weight));
    }
  }

  [[nodiscard]] constexpr int Rank() const noexcept { return rank_; }
  [[nodiscard]] constexpr int Weight() const noexcept { return weight_; }
  [[nodiscard]] constexpr bool Voigt() const noexcept { return voigt_; }

  /// Whether the stored components are a Voigt vector or matrix: ranks 2
  /// and 4 with `voigt`.
  [[nodiscard]] bool HasVoigtShape() const noexcept {
    return voigt_ && (rank_ == 2 || rank_ == 4);
  }

  /// The trailing shape of a stored value.
  [[nodiscard]] std::vector<std::size_t> StoredShape() const {
    if (HasVoigtShape()) {
      return rank_ == 2 ? std::vector<std::size_t>{6} : std::vector<std::size_t>{6, 6};
    }
    return std::vector<std::size_t>(static_cast<std::size_t>(rank_), 3);
  }

  /// The number of stored components.
  [[nodiscard]] std::size_t StoredSize() const {
    auto n = std::size_t{1};
    for (const auto extent : StoredShape()) n *= extent;
    return n;
  }

  friend bool operator==(const Character&, const Character&) = default;

 private:
  int rank_;
  int weight_;
  bool voigt_;
};

/// Q_kappa, Q_mu, eta, relaxation times, viscosity, velocities.
inline constexpr auto SCALAR = Character(0, 0);
/// Density, and every modulus, which shares its factor of 1/J.
inline constexpr auto DENSITY = Character(0, 1);
/// A vector.
inline constexpr auto VECTOR = Character(1, 0);
/// Equilibrium stress, stress glut.
inline constexpr auto STRESS = Character(2, 1);
/// The second elasticity tensor.
inline constexpr auto ELASTIC = Character(4, 1);

}  // namespace planetmodel

#endif  // PLANETMODEL_CHARACTER_HPP
