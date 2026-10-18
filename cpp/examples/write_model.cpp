// Write a model file of fields of one's own: a mantle of two layers, a
// density of the radius alone and a viscosity that varies with direction,
// and the mapping a flattening of the surface, linear in r below it.
//
//   planetmodel_write_model out.nc
#include <planetmodel/modelfile.hpp>

#include <cmath>
#include <iostream>
#include <optional>

int main(int argc, char** argv) {
  namespace pm = planetmodel;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " out.nc\n";
    return 2;
  }
  try {
    auto file = pm::ModelFile(pm::Skeleton({3480e3, 5701e3, 6371e3}));
    file.SetName("example mantle");
    file.SetLayerName(0, "lower mantle");
    file.SetLayerName(1, "upper mantle");
    file.SetInterfaceName(0, "cmb");
    file.SetInterfaceName(2, "surface");

    // one part, both layers on one grid, elements of 200 km or less
    const auto& skeleton = file.GetSkeleton();
    file.AddPart(0, 1, pm::ElementEdges(skeleton, 0, 1, 200e3), 5, pm::GaussLegendre(16));

    // the density, a vocabulary name, of the radius alone
    auto rho = [](double r) { return 5600.0 - 0.0004 * (r - 3480e3); };
    file.AddField("rho", {{0, pm::LayerSource::Radial(rho)}, {1, pm::LayerSource::Radial(rho)}});

    // the viscosity, varying with direction in the upper mantle, read
    // between the grid's nodes by the bicubic rule
    file.AddField(
        "viscosity",
        {{0, pm::LayerSource::Radial([](double) { return 1e22; })},
         {1, pm::LayerSource::Lateral([](double r, double theta, double phi) {
            const auto depth = (6371e3 - r) / 1e3;
            return 1e21 * std::exp(0.004 * depth + 0.5 * std::sin(theta) * std::cos(phi));
          })}},
        {.lateral = pm::LateralRule::Cubic});

    // the surface moved by a flattening, the core-mantle boundary still
    const auto flattening = 1.0 / 298.25;
    file.SetStack({3480e3, 6371e3}, pm::GaussLegendre(8),
                  {std::nullopt, pm::LayerSource::Lateral([=](double r, double theta, double) {
                     const auto c = std::cos(theta);
                     return -flattening * r * (c * c - 1.0 / 3.0);
                   })});

    pm::WriteModelFile(argv[1], file);
    std::cout << "wrote " << argv[1] << "\n";
  } catch (const pm::Error& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
