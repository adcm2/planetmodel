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
#include <numbers>

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
