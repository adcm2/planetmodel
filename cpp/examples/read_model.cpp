// Read a model file: what it holds on each layer, every scalar field's
// value and gradient at a point of its layer, the mapping's displacement
// of that point, and every field on an interface at a point of it.
//
//   planetmodel_read_model model.nc
#include <planetmodel/modelfile.hpp>

#include <format>
#include <iostream>

int main(int argc, char** argv) {
  namespace pm = planetmodel;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " model.nc\n";
    return 2;
  }
  try {
    const auto file = pm::ReadModelFile(argv[1]);
    std::cout << std::format("{}: {}, schema {}, length scale {} m\n", file.Source(),
                             file.Name().value_or("unnamed"), file.Schema(),
                             file.GetScales().Length());
    const auto mapping = file.Mapping();
    const auto& skeleton = file.GetSkeleton();
    for (auto layer = 0; layer < static_cast<int>(skeleton.NLayers()); ++layer) {
      const auto [lo, hi] = skeleton.Interval(static_cast<std::size_t>(layer));
      const auto name = file.LayerNames()[static_cast<std::size_t>(layer)].value_or("-");
      std::cout << std::format("layer {} ({}), r in [{}, {}]:\n", layer, name, lo, hi);
      // a point of the layer, its colatitude and longitude in radians
      const auto r = 0.5 * (lo + hi);
      const auto theta = 1.0;
      const auto phi = 2.0;
      for (const auto& field_name : file.Names(layer)) {
        const auto field = file.FieldOn(layer, field_name);
        if (field.GetCharacter().Rank() != 0) {
          std::cout << std::format("  {}: rank {}, {} components\n", field_name,
                                   field.GetCharacter().Rank(), field.Size());
          continue;
        }
        const auto scalar = field.AsScalar();
        const auto [dr, dtheta, dphi] = scalar.Gradient(r, theta, phi);
        std::cout << std::format("  {}{}: {:.6g}, gradient ({:.4g}, {:.4g}, {:.4g})\n",
                                 field_name, field.IsRadial() ? " (of r)" : "",
                                 scalar.Value(r, theta, phi), dr, dtheta, dphi);
      }
      const auto u = mapping.Displacement(pm::CartesianPoint(r, theta, phi));
      std::cout << std::format("  displacement at the point: ({:.4g}, {:.4g}, {:.4g})\n", u[0],
                               u[1], u[2]);
    }
    for (auto k = 0; k < static_cast<int>(file.InterfaceRadii().size()); ++k) {
      const auto name = file.InterfaceNames()[static_cast<std::size_t>(k)].value_or("-");
      for (const auto& field_name : file.SurfaceNames(k)) {
        const auto f = file.SurfaceFieldOn(k, field_name);
        const auto [dtheta, dphi] = f.Gradient(1.0, 2.0);
        std::cout << std::format("interface {} ({}), {}: {:.6g}, gradient ({:.4g}, {:.4g})\n",
                                 k, name, field_name, f.Value(1.0, 2.0), dtheta, dphi);
      }
    }
  } catch (const pm::Error& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
