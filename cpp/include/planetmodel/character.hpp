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
  explicit Character(int rank, int weight = 0, bool voigt = true)
      : rank_(rank), weight_(weight), voigt_(voigt) {
    if (rank < 0) {
      throw Error("Character", "", std::format("rank must be non-negative, got {}", rank));
    }
    if (weight != 0 && weight != 1) {
      throw Error("Character", "", std::format("weight must be 0 or 1, got {}", weight));
    }
  }

  [[nodiscard]] int Rank() const noexcept { return rank_; }
  [[nodiscard]] int Weight() const noexcept { return weight_; }
  [[nodiscard]] bool Voigt() const noexcept { return voigt_; }

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

}  // namespace planetmodel

#endif  // PLANETMODEL_CHARACTER_HPP
