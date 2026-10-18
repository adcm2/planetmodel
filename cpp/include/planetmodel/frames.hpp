#ifndef PLANETMODEL_FRAMES_HPP
#define PLANETMODEL_FRAMES_HPP

/**
 * @file
 * @brief The local spherical frame and the reduction of angles, as
 * `planetmodel.frames` defines them.
 *
 * Components are in the frame (e_r, e_theta, e_phi) at (theta, phi),
 * theta the colatitude: the frame matrix R holds the three as its columns
 * in Cartesian components, so that Cartesian = R spherical and
 * spherical = R^T Cartesian.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace planetmodel {

/** @brief A vector's three components. */
using Vector3 = std::array<double, 3>;

/** @brief A 3 by 3 matrix, by rows: `m[i][j]`. */
using Matrix3 = std::array<std::array<double, 3>, 3>;

/**
 * @brief The frame matrix at a point: column 0 is e_r, 1 is e_theta and
 * 2 is e_phi, each in Cartesian components.
 * @param theta The colatitude.
 * @param phi The longitude.
 */
[[nodiscard]] inline Matrix3 SphericalFrame(double theta, double phi) noexcept {
  const auto st = std::sin(theta);
  const auto ct = std::cos(theta);
  const auto sp = std::sin(phi);
  const auto cp = std::cos(phi);
  return {{{st * cp, ct * cp, -sp}, {st * sp, ct * sp, cp}, {ct, -st, 0.0}}};
}

/**
 * @brief Spherical components at a point as Cartesian ones: R v.
 * @param v The components in the frame (e_r, e_theta, e_phi).
 * @param theta The colatitude.
 * @param phi The longitude.
 */
[[nodiscard]] inline Vector3 ToCartesian(const Vector3& v, double theta, double phi) noexcept {
  const auto R = SphericalFrame(theta, phi);
  auto out = Vector3{};
  for (auto i = 0; i < 3; ++i) {
    for (auto k = 0; k < 3; ++k) out[i] += R[i][k] * v[k];
  }
  return out;
}

/**
 * @brief Cartesian components as spherical ones at a point: R^T v.
 * @param v The Cartesian components.
 * @param theta The colatitude.
 * @param phi The longitude.
 */
[[nodiscard]] inline Vector3 ToSpherical(const Vector3& v, double theta, double phi) noexcept {
  const auto R = SphericalFrame(theta, phi);
  auto out = Vector3{};
  for (auto k = 0; k < 3; ++k) {
    for (auto i = 0; i < 3; ++i) out[k] += R[i][k] * v[i];
  }
  return out;
}

/**
 * @brief The derivatives of the frame matrix in colatitude and longitude.
 *
 * d/dtheta takes e_r to e_theta and e_theta to -e_r; d/dphi takes e_r to
 * sin(theta) e_phi, e_theta to cos(theta) e_phi and e_phi to
 * -(sin(theta) e_r + cos(theta) e_theta).
 *
 * @param theta The colatitude.
 * @param phi The longitude.
 * @return (dR/dtheta, dR/dphi), columns as in `SphericalFrame`.
 */
[[nodiscard]] inline std::array<Matrix3, 2> SphericalFrameDerivatives(double theta,
                                                                     double phi) noexcept {
  const auto st = std::sin(theta);
  const auto ct = std::cos(theta);
  const auto sp = std::sin(phi);
  const auto cp = std::cos(phi);
  const auto dtheta = Matrix3{{{ct * cp, -st * cp, 0.0}, {ct * sp, -st * sp, 0.0},
                               {-st, -ct, 0.0}}};
  const auto dphi = Matrix3{{{-st * sp, -ct * sp, -cp}, {st * cp, ct * cp, -sp},
                             {0.0, 0.0, 0.0}}};
  return {dtheta, dphi};
}

/**
 * @brief The index pairs behind the six Voigt slots, in Voigt order
 * (11, 22, 33, 23, 13, 12), counted from zero.
 */
inline constexpr auto VOIGT_PAIRS = std::array<std::array<int, 2>, 6>{
    {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}}};

/**
 * @brief The Voigt slot of an index pair.
 * @param i The first index, 0 to 2.
 * @param j The second index, 0 to 2.
 */
[[nodiscard]] constexpr int VoigtSlot(int i, int j) noexcept {
  if (i == j) return i;
  const auto k = i + j;  // 1 for (0, 1), 2 for (0, 2), 3 for (1, 2)
  return k == 1 ? 5 : (k == 2 ? 4 : 3);
}

/**
 * @brief The components of a stored value, every factor of its rank
 * turned by its own matrix: out_{i...} = M1_{ia} M2_{jb} ... T_{ab...}.
 *
 * The stored components are a scalar, a vector, a Voigt vector or matrix
 * (ranks 2 and 4, no engineering factors) or the full 3 per index, in C
 * order; a Voigt value is turned as the symmetric tensor it stands for.
 * With every matrix R this takes spherical components to Cartesian ones,
 * with R^T back; with one factor a derivative of R it differentiates a
 * radial field's Cartesian components.
 *
 * @param stored The stored components, `rank` and `voigt` saying their
 *        layout.
 * @param rank The rank.
 * @param voigt Whether a rank-2 or rank-4 value is in Voigt form.
 * @param matrices One matrix per factor, `rank` of them.
 * @return The turned components, in the same layout.
 */
[[nodiscard]] inline std::vector<double> RotateStored(std::span<const double> stored, int rank,
                                                      bool voigt,
                                                      std::span<const Matrix3> matrices) {
  if (rank == 0) return {stored.begin(), stored.end()};
  const auto is_voigt = voigt && (rank == 2 || rank == 4);
  auto size = std::size_t{1};
  for (auto s = 0; s < rank; ++s) size *= 3;
  // the full tensor, C order
  auto full = std::vector<double>(size);
  for (auto flat = std::size_t{0}; flat < size; ++flat) {
    if (!is_voigt) {
      full[flat] = stored[flat];
      continue;
    }
    auto index = std::array<int, 4>{};
    auto rest = flat;
    for (auto s = rank - 1; s >= 0; --s) {
      index[static_cast<std::size_t>(s)] = static_cast<int>(rest % 3);
      rest /= 3;
    }
    full[flat] = rank == 2 ? stored[static_cast<std::size_t>(VoigtSlot(index[0], index[1]))]
                           : stored[static_cast<std::size_t>(6 * VoigtSlot(index[0], index[1]) +
                                                             VoigtSlot(index[2], index[3]))];
  }
  // one factor at a time
  auto stride = size;
  for (auto s = 0; s < rank; ++s) {
    stride /= 3;
    const auto& M = matrices[static_cast<std::size_t>(s)];
    auto next = std::vector<double>(size, 0.0);
    for (auto flat = std::size_t{0}; flat < size; ++flat) {
      const auto here = static_cast<int>((flat / stride) % 3);
      const auto base = flat - static_cast<std::size_t>(here) * stride;
      for (auto a = 0; a < 3; ++a) {
        next[flat] += M[static_cast<std::size_t>(here)][static_cast<std::size_t>(a)] *
                      full[base + static_cast<std::size_t>(a) * stride];
      }
    }
    full = std::move(next);
  }
  if (!is_voigt) return full;
  auto out = std::vector<double>(rank == 2 ? 6 : 36);
  for (auto a = 0; a < 6; ++a) {
    const auto [i, j] = VOIGT_PAIRS[static_cast<std::size_t>(a)];
    if (rank == 2) {
      out[static_cast<std::size_t>(a)] = full[static_cast<std::size_t>(3 * i + j)];
      continue;
    }
    for (auto b = 0; b < 6; ++b) {
      const auto [k, l] = VOIGT_PAIRS[static_cast<std::size_t>(b)];
      out[static_cast<std::size_t>(6 * a + b)] =
          full[static_cast<std::size_t>(((3 * i + j) * 3 + k) * 3 + l)];
    }
  }
  return out;
}

/**
 * @brief An angle reduced into [start, start + period) by whole periods.
 *
 * As numpy's mod reduces it: a small negative offset that rounds up to the
 * period itself is brought back to `start`.
 *
 * @param x The angle.
 * @param start The start of the range.
 * @param period The period, a full turn by default.
 */
[[nodiscard]] inline double Wrapped(double x, double start = 0.0,
                                    double period = 2.0 * std::numbers::pi) noexcept {
  auto d = std::fmod(x - start, period);
  if (d != 0.0 && d < 0.0) d += period;
  if (d >= period) d -= period;
  return start + d;
}

}  // namespace planetmodel

#endif  // PLANETMODEL_FRAMES_HPP
