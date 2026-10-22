#include "planetmodel/radial_model.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "planetmodel/skeleton.hpp"

namespace {

planetmodel::Deck IsotropicDeck(double tref = 10.0, bool fluid = false) {
  const auto header = std::vector<std::string>{"two layer test", "0 " + std::to_string(tref) + " 1",
                                               "6 0 0"};
  return planetmodel::Deck(
      {0.0, 0.5, 1.0, 1.0, 1.5, 2.0},
      {{"rho", {2, 3, 4, 5, 6, 7}}, {"vp", {3, 4, 5, 6, 7, 8}},
       {"vs", {fluid ? 0.0 : 1.0, fluid ? 0.0 : 1.5, fluid ? 0.0 : 2.0,
                2.0, 2.5, 3.0}},
       {"qkappa", {50, 50, 50, 50, 50, 50}}, {"qmu", {0, 0, 0, 0, 0, 0}}},
      header, "synthetic mineos");
}

TEST(RadialModel, MineosPreservesLayersInterfacesMetadataAndKnots) {
  const auto model = planetmodel::MineosModel(IsotropicDeck());
  EXPECT_EQ(model.GetSkeleton().Boundaries(), (std::vector<double>{0, 1, 2}));
  ASSERT_EQ(model.Layers().size(), 2u);
  EXPECT_EQ(model.Layer(0).fields.at("rho").Value(1.0), 4.0);
  EXPECT_EQ(model.Layer(1).fields.at("rho").Value(1.0), 5.0);
  EXPECT_EQ(model.InterfaceNames().size(), 2u);
  EXPECT_EQ(model.InterfaceNames().back(), "surface");
  ASSERT_TRUE(model.OmegaRef());
  EXPECT_NEAR(*model.OmegaRef(), 2.0 * std::acos(-1.0) / 10.0, 1e-14);
  EXPECT_DOUBLE_EQ(model.G(), planetmodel::G_SI);
  EXPECT_TRUE(model.GetScales().IsSI());
  EXPECT_EQ(model.Input(), "synthetic mineos");
  EXPECT_EQ(model.IntegrationBreakpoints(0), (std::vector<double>{0, 0.5, 1}));
  EXPECT_EQ(model.IntegrationBreakpoints(1), (std::vector<double>{1, 1.5, 2}));
  EXPECT_FALSE(model.Material(0).IsFluid(0.5));  // qmu=0 means lossless, not fluid.
  EXPECT_FALSE(model.IsFluid(0));
}

TEST(RadialModel, ConvertsIsotropicVelocitiesPointwiseWithDerivatives) {
  const auto model = planetmodel::MineosModel(IsotropicDeck(0.0));
  const auto material = model.Material(0);
  const auto mu = material.Evaluate("mu", 0.5);
  const auto kappa = material.Evaluate("kappa", 0.5);
  EXPECT_DOUBLE_EQ(mu.value, 6.75);
  EXPECT_DOUBLE_EQ(mu.derivative, 13.5);
  EXPECT_DOUBLE_EQ(kappa.value, 39.0);
  EXPECT_DOUBLE_EQ(kappa.derivative, 62.0);
  EXPECT_FALSE(model.OmegaRef());  // Elastic models need no attenuation reference.
}

TEST(RadialModel, ConvertsTransverseIsotropyAndDetectsFluidFromShear) {
  const auto deck = planetmodel::Deck(
      {0, 1}, {{"rho", {4, 4}}, {"vpv", {3, 3}}, {"vsv", {0, 0}},
               {"qkappa", {60, 60}}, {"qmu", {0, 0}}, {"vph", {4, 4}},
               {"vsh", {0, 0}}, {"eta", {1.5, 1.5}}},
      {"TI", "1 5 1", "2 0 0"}, "ti");
  const auto material = planetmodel::MineosModel(deck).Material(0);
  EXPECT_DOUBLE_EQ(material.Evaluate("A", 0.5).value, 64.0);
  EXPECT_DOUBLE_EQ(material.Evaluate("C", 0.5).value, 36.0);
  EXPECT_DOUBLE_EQ(material.Evaluate("F", 0.5).value, 96.0);
  EXPECT_TRUE(material.IsFluid(0.5));
  EXPECT_TRUE(planetmodel::MineosModel(IsotropicDeck(10.0, true)).IsFluid(0));
  EXPECT_FALSE(planetmodel::MineosModel(IsotropicDeck(10.0, true)).IsFluid(1));
}

TEST(RadialModel, GenericConstructionUsesModelUnitsAndEndpointBreakpoints) {
  const auto scales = planetmodel::Scales::Geophysical(2.0, 5.0);
  std::vector<planetmodel::RadialLayer> layers(1);
  layers[0].fields.emplace("rho", planetmodel::ThroughKnots(
      {0.0, 2.0}, {4.0, 4.0}, planetmodel::KnotKind::Linear, 0.0));
  const auto model = planetmodel::RadialModel(planetmodel::Skeleton({0.0, 2.0}),
                                               std::move(layers), scales);
  EXPECT_DOUBLE_EQ(model.G(), 1.0);
  EXPECT_EQ(model.IntegrationBreakpoints(0), (std::vector<double>{0.0, 2.0}));
  EXPECT_DOUBLE_EQ(model.Layer(0).fields.at("rho").Value(1.0), 4.0);
}


TEST(RadialModel, CanonicalDescriptionWinsOverPartialOrRedundantFields) {
  const auto field = [](double value) {
    return planetmodel::ThroughKnots({0, 1}, {value, value}, planetmodel::KnotKind::Linear, 0);
  };
  std::map<std::string, planetmodel::RadialFunction> from_velocity{
      {"rho", field(2)}, {"vp", field(4)}, {"vs", field(1)}, {"A", field(999)}};
  const auto velocity_material = planetmodel::RadialMaterial(std::move(from_velocity));
  EXPECT_DOUBLE_EQ(velocity_material.Evaluate("A", 0.5).value, 32.0);

  std::map<std::string, planetmodel::RadialFunction> from_love{
      {"A", field(10)}, {"C", field(10)}, {"F", field(6)}, {"L", field(2)},
      {"N", field(2)}, {"kappa", field(100)}, {"mu", field(200)}};
  const auto love_material = planetmodel::RadialMaterial(std::move(from_love));
  EXPECT_NEAR(love_material.Evaluate("kappa", 0.5).value, 66.0 / 9.0, 1e-14);
  EXPECT_DOUBLE_EQ(love_material.Evaluate("mu", 0.5).value, 2.0);

  std::map<std::string, planetmodel::RadialFunction> conflicting_shear{
      {"A", field(10)}, {"C", field(10)}, {"F", field(10)}, {"L", field(0)},
      {"N", field(0)}, {"vs", field(1)}};
  EXPECT_FALSE(planetmodel::RadialMaterial(std::move(conflicting_shear)).IsFluid(0.5));
}

TEST(RadialModel, GenericFluidClassificationRequiresAnExplicitState) {
  std::vector<planetmodel::RadialLayer> layers(1);
  layers[0].fields.emplace("rho", planetmodel::ThroughKnots(
      {0, 1}, {1, 1}, planetmodel::KnotKind::Linear, 0));
  auto model = planetmodel::RadialModel(planetmodel::Skeleton({0, 1}), layers);
  EXPECT_THROW(static_cast<void>(model.IsFluid(0)), planetmodel::Error);
  layers[0].shear_state = planetmodel::ShearState::Fluid;
  model = planetmodel::RadialModel(planetmodel::Skeleton({0, 1}), std::move(layers));
  EXPECT_TRUE(model.IsFluid(0));
}

TEST(RadialModel, RefusesInvalidQualityFactorsAndPartialModuli) {
  auto columns = std::vector<std::pair<std::string, std::vector<double>>>{
      {"rho", {2, 2}}, {"vp", {3, 3}}, {"vs", {1, 1}}, {"qkappa", {-1, 2}}, {"qmu", {0, 0}}};
  const auto invalid = planetmodel::Deck({0, 1}, std::move(columns),
                                         {"bad", "0 1 1", "2 0 0"}, "bad-q");
  EXPECT_THROW(static_cast<void>(planetmodel::MineosModel(invalid)), planetmodel::Error);
  std::map<std::string, planetmodel::RadialFunction> partial;
  partial.emplace("A", planetmodel::ThroughKnots({0, 1}, {1, 1}, planetmodel::KnotKind::Linear, 0));
  EXPECT_THROW(static_cast<void>(planetmodel::RadialMaterial(std::move(partial))), planetmodel::Error);
}

}  // namespace
