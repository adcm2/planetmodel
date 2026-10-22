#include "planetmodel/gravity.hpp"
#include "planetmodel/rheology.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numbers>
#include <string>
#include <vector>

namespace {

planetmodel::RadialFunction Constant(double lo, double hi, double value) {
  return {lo, hi, 0.0, [value](double, int d) { return d == 0 ? value : 0.0; }};
}

TEST(RadialGravity, UniformSphereAndModelUnits) {
  constexpr double radius = 2.0, rho = 3.0, G = 5.0;
  planetmodel::RadialLayer layer;
  layer.fields.emplace("rho", Constant(0.0, radius, rho));
  const planetmodel::RadialModel model(planetmodel::Skeleton({0.0, radius}), {layer},
                                        planetmodel::Scales::Geophysical(1.0, 1.0), G);
  const planetmodel::RadialGravity gravity(model);
  const auto r = 0.7;
  EXPECT_NEAR(gravity.Mass(0, r), 4.0 * std::numbers::pi * rho * std::pow(r, 3) / 3.0, 1e-13);
  EXPECT_NEAR(gravity.Gravity(0, r), 4.0 * std::numbers::pi * G * rho * r / 3.0, 1e-13);
  EXPECT_NEAR(gravity.GravityDerivative(0, r), 4.0 * std::numbers::pi * G * rho / 3.0, 1e-13);
  EXPECT_DOUBLE_EQ(gravity.Gravity(0, 0.0), 0.0);
  EXPECT_NEAR(gravity.GravityDerivative(0, 0.0), 4.0 * std::numbers::pi * G * rho / 3.0, 1e-13);
}

TEST(RadialGravity, LayerSidesAndHollowCavity) {
  planetmodel::RadialLayer inner, outer;
  inner.fields.emplace("rho", Constant(1.0, 2.0, 2.0));
  outer.fields.emplace("rho", Constant(2.0, 3.0, 3.0));
  const planetmodel::RadialModel layered(planetmodel::Skeleton({1.0, 2.0, 3.0}),
                                          {inner, outer}, planetmodel::Scales::SI(), 1.0);
  const planetmodel::RadialGravity gravity(layered);
  const auto expected_mass = 4.0 * std::numbers::pi / 3.0 * 2.0 * (8.0 - 1.0);
  EXPECT_NEAR(gravity.Mass(0, 2.0), expected_mass, 1e-12);
  EXPECT_DOUBLE_EQ(gravity.Gravity(0, 2.0), gravity.Gravity(1, 2.0));
  EXPECT_NE(gravity.GravityDerivative(0, 2.0), gravity.GravityDerivative(1, 2.0));

  planetmodel::RadialLayer shell;
  shell.fields.emplace("rho", Constant(1.0, 2.0, 4.0));
  const planetmodel::RadialModel hollow(planetmodel::Skeleton({1.0, 2.0}), {shell});
  const planetmodel::RadialGravity hollow_gravity(hollow);
  EXPECT_DOUBLE_EQ(hollow_gravity.Mass(0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(hollow_gravity.Mass(0.5), 0.0);
  EXPECT_DOUBLE_EQ(hollow_gravity.Gravity(0.5), 0.0);
  EXPECT_DOUBLE_EQ(hollow_gravity.GravityDerivative(0.5), 0.0);
  EXPECT_DOUBLE_EQ(hollow_gravity.Gravity(0, 1.0), 0.0);
  EXPECT_NEAR(hollow_gravity.GravityDerivative(0, 1.0), 4.0 * std::numbers::pi *
              planetmodel::G_SI * 4.0, 1e-20);
  EXPECT_THROW(static_cast<void>(hollow_gravity.Gravity(0, 0.5)), planetmodel::Error);
}

TEST(RadialGravity, AdaptiveIntegralRespectsBreakpointsAndConvergenceFailure) {
  constexpr double a = 2.0;
  planetmodel::RadialLayer layer;
  layer.fields.emplace("rho", planetmodel::RadialFunction(0.0, 1.0, 0.0,
      [](double r, int d) { return d == 0 ? std::exp(a * r) : a * std::exp(a * r); }));
  const planetmodel::RadialModel model(planetmodel::Skeleton({0.0, 1.0}), {layer},
      planetmodel::Scales::SI(), 1.0, {}, {}, {}, {{0.0, 0.4, 1.0}});
  const planetmodel::RadialGravity gravity(model);
  const auto r = 0.73;
  const auto integral = std::exp(a * r) * (r * r / a - 2.0 * r / (a * a) + 2.0 / (a * a * a)) -
                        2.0 / (a * a * a);
  EXPECT_NEAR(gravity.Mass(0, r), 4.0 * std::numbers::pi * integral, 1e-11);
  EXPECT_EQ(model.IntegrationBreakpoints(0), (std::vector<double>{0.0, 0.4, 1.0}));

  planetmodel::RadialLayer hard_layer;
  hard_layer.fields.emplace("rho", planetmodel::RadialFunction(0.0, 1.0, 0.0,
      [](double x, int d) { return d == 0 ? std::exp(100.0 * x) : 100.0 * std::exp(100.0 * x); }));
  const planetmodel::RadialModel hard_model(planetmodel::Skeleton({0.0, 1.0}), {hard_layer});
  const auto strict = planetmodel::GravityIntegrationSettings{.relative_tolerance = 1e-12,
      .absolute_tolerance = 0.0, .max_depth = 1};
  EXPECT_THROW(static_cast<void>(planetmodel::RadialGravity(hard_model, strict)), planetmodel::Error);

  planetmodel::RadialLayer negative;
  negative.fields.emplace("rho", Constant(0.0, 1.0, -1.0));
  EXPECT_THROW(static_cast<void>(planetmodel::RadialGravity(
      planetmodel::RadialModel(planetmodel::Skeleton({0.0, 1.0}), {negative}))), planetmodel::Error);
  planetmodel::RadialLayer center_nan;
  center_nan.fields.emplace("rho", planetmodel::RadialFunction(0.0, 1.0, 0.0,
      [](double x, int) { return x == 0.0 ? std::numeric_limits<double>::quiet_NaN() : 1.0; }));
  const planetmodel::RadialGravity nan_center(
      planetmodel::RadialModel(planetmodel::Skeleton({0.0, 1.0}), {center_nan}));
  EXPECT_THROW(static_cast<void>(nan_center.GravityDerivative(0, 0.0)), planetmodel::Error);
  const auto tiny_budget = planetmodel::GravityIntegrationSettings{.max_evaluations = 1};
  EXPECT_THROW(static_cast<void>(planetmodel::RadialGravity(model, tiny_budget)), planetmodel::Error);
}

planetmodel::RadialMaterial IsotropicMaterial(double qkappa, double qmu) {
  std::map<std::string, planetmodel::RadialFunction> fields{
      {"rho", Constant(0.0, 1.0, 4.0)}, {"vp", Constant(0.0, 1.0, 3.0)},
      {"vs", Constant(0.0, 1.0, 1.0)}, {"qkappa", Constant(0.0, 1.0, qkappa)},
      {"qmu", Constant(0.0, 1.0, qmu)}};
  return planetmodel::RadialMaterial(std::move(fields));
}

TEST(PreparedConstantQ, IsotropicVoigtLossAndElasticLimit) {
  const auto material = IsotropicMaterial(2.0, 4.0);
  const planetmodel::PreparedConstantQ prepared(material, 0.5, 3.0);
  const auto at_ref = prepared.Evaluate({3.0, 0.0});
  EXPECT_NEAR(at_ref.A.real(), 36.0, 1e-13);
  EXPECT_NEAR(at_ref.A.imag(), 92.0 / 3.0 / 2.0 + 4.0 / 3.0, 1e-13);
  EXPECT_NEAR(at_ref.F.imag(), 92.0 / 3.0 / 2.0 - 2.0 / 3.0, 1e-13);
  EXPECT_NEAR(at_ref.L.imag(), 1.0, 1e-13);
  EXPECT_NEAR(at_ref.N.imag(), 1.0, 1e-13);

  const auto plus = prepared.Evaluate({6.0, 0.0});
  const auto minus = prepared.Evaluate({-6.0, 0.0});
  EXPECT_NEAR(std::abs(minus.A - std::conj(plus.A)), 0.0, 1e-13);
  EXPECT_NEAR(std::abs(minus.C - std::conj(plus.C)), 0.0, 1e-13);
  EXPECT_NEAR(std::abs(minus.F - std::conj(plus.F)), 0.0, 1e-13);
  EXPECT_NEAR(std::abs(minus.L - std::conj(plus.L)), 0.0, 1e-13);
  EXPECT_NEAR(std::abs(minus.N - std::conj(plus.N)), 0.0, 1e-13);
  EXPECT_NEAR(std::abs(planetmodel::ConstantQFactor({2.0, -0.5}, 3.0) -
                       planetmodel::ConstantQFactor({10.0, -2.5}, 15.0)), 0.0, 1e-14);
  const auto at_twice_ref = prepared.Evaluate({6.0, 0.0});
  const auto log_factor = (2.0 / std::numbers::pi) * std::log(2.0);
  const auto want_A_real = 36.0 + (46.0 / 3.0 + 4.0 / 3.0) * log_factor;
  EXPECT_NEAR(at_twice_ref.A.real(), want_A_real, 1e-13);
  const auto damped = prepared.Evaluate({2.0, -1.0});
  const auto damped_real = (2.0 / std::numbers::pi) * std::log(std::sqrt(5.0) / 3.0);
  const auto damped_imag = (2.0 / std::numbers::pi) * std::atan2(2.0, 1.0);
  EXPECT_NEAR(damped.A.real(), 36.0 + (46.0 / 3.0 + 4.0 / 3.0) * damped_real, 1e-13);
  EXPECT_NEAR(damped.A.imag(), (46.0 / 3.0 + 4.0 / 3.0) * damped_imag, 1e-13);
  const auto purely_damped = planetmodel::ConstantQFactor({0.0, -1.5}, 3.0);
  EXPECT_NEAR(purely_damped.real(), (2.0 / std::numbers::pi) * std::log(0.5), 1e-14);
  EXPECT_NEAR(purely_damped.imag(), 0.0, 1e-14);

  const auto elastic = planetmodel::PreparedConstantQ(IsotropicMaterial(0.0, 0.0), 0.5);
  EXPECT_FALSE(elastic.HasAttenuation());
  const auto unchanged = elastic.Evaluate({1.0, 0.0});
  EXPECT_DOUBLE_EQ(unchanged.A.real(), 36.0);
  EXPECT_DOUBLE_EQ(unchanged.A.imag(), 0.0);
}

TEST(PreparedConstantQ, PreservesTransverseIsotropyAndRejectsInvalidInputs) {
  std::map<std::string, planetmodel::RadialFunction> fields{
      {"rho", Constant(0, 1, 4)}, {"vpv", Constant(0, 1, 3)},
      {"vph", Constant(0, 1, 4)}, {"vsv", Constant(0, 1, 1)},
      {"vsh", Constant(0, 1, 2)}, {"eta", Constant(0, 1, 1.5)},
      {"qkappa", Constant(0, 1, 10)}, {"qmu", Constant(0, 1, 20)}};
  const auto prepared = planetmodel::PreparedConstantQ(planetmodel::RadialMaterial(fields), 0.5, 2.0);
  const auto moduli = prepared.Evaluate({2.0, 0.0});
  const auto base = prepared.Elastic();
  EXPECT_NEAR(moduli.A.real() - moduli.C.real(), base.A - base.C, 1e-13);
  EXPECT_NEAR(moduli.A.imag() - moduli.C.imag(), 0.0, 1e-13);
  EXPECT_THROW(static_cast<void>(prepared.Evaluate({1.0, 0.1})), planetmodel::Error);
  EXPECT_THROW(static_cast<void>(prepared.Evaluate({0.0, 0.0})), planetmodel::Error);
  EXPECT_THROW(static_cast<void>(planetmodel::ConstantQFactor({1.0, 0.0}, 0.0)), planetmodel::Error);

  auto bad_q = IsotropicMaterial(0.0, -1.0);
  EXPECT_THROW(static_cast<void>(planetmodel::PreparedConstantQ(bad_q, 0.5, 1.0)), planetmodel::Error);
  auto unsupported_fields = IsotropicMaterial(1.0, 1.0).Fields();
  unsupported_fields.emplace("viscosity", Constant(0, 1, 2));
  EXPECT_THROW(static_cast<void>(planetmodel::PreparedConstantQ(
      planetmodel::RadialMaterial(std::move(unsupported_fields)), 0.5, 1.0)), planetmodel::Error);
  EXPECT_THROW(static_cast<void>(planetmodel::PreparedConstantQ(IsotropicMaterial(1, 0), 0.5)),
               planetmodel::Error);
  std::map<std::string, planetmodel::RadialFunction> fluid_fields{
      {"rho", Constant(0, 1, 4)}, {"vp", Constant(0, 1, 3)},
      {"vs", Constant(0, 1, 0)}, {"qmu", Constant(0, 1, 100)}};
  const auto fluid = planetmodel::PreparedConstantQ(
      planetmodel::RadialMaterial(std::move(fluid_fields)), 0.5);
  EXPECT_FALSE(fluid.HasAttenuation());
  EXPECT_DOUBLE_EQ(fluid.Evaluate({2.0, 0.0}).A.imag(), 0.0);
}

}  // namespace
