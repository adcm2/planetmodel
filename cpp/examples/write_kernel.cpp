// Write the file a code that computes sensitivity kernels writes: the
// derivative of an objective with respect to the shear velocity, per unit
// volume, on the reference skeleton of the model the code ran on, at the
// code's own radial nodes (elements of two nodes, so that its values are
// kept as they are, linear between) and on a Gauss-Legendre grid, read
// between its nodes by the bilinear rule, which never overshoots: the peak
// is narrow beside the grid's spacing, and the bicubic rule would ring
// around it with values of the wrong sign; and the
// derivative with respect to a shift of the 670 km discontinuity, per unit
// area, on that interface.  The kernels are synthetic: a broad pattern that
// changes sign, and a peak beside a receiver at the surface, as a
// numerical kernel has.  The code names its kernels and states their
// dimensions; the library holds no names of kernels.
//
//   planetmodel_write_kernel kernel.nc
#include <planetmodel/modelfile.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>

namespace {

constexpr auto pi = std::numbers::pi;
constexpr auto surface = 6371e3;
constexpr auto cmb = 3480e3;
constexpr auto d670 = 5701e3;

/// The receiver: at the surface, colatitude 60 degrees, longitude 30.
constexpr auto receiver_theta = pi / 3.0;
constexpr auto receiver_phi = pi / 6.0;

/// The angle between a direction and the receiver's.
double FromReceiver(double theta, double phi) {
  const auto c = std::cos(theta) * std::cos(receiver_theta) +
                 std::sin(theta) * std::sin(receiver_theta) * std::cos(phi - receiver_phi);
  return std::acos(std::clamp(c, -1.0, 1.0));
}

/// The kernel of the shear velocity: a pattern of two signs, zero at the
/// core-mantle boundary and the surface, and a peak 200 km wide at the
/// receiver.
double ShearKernel(double r, double theta, double phi) {
  const auto s = (r - cmb) / (surface - cmb);
  const auto delta = FromReceiver(theta, phi);
  const auto broad = std::sin(pi * s) * std::cos(2.0 * delta);
  // the squared distance from the receiver, by the law of cosines
  const auto d2 = r * r + surface * surface - 2.0 * r * surface * std::cos(delta);
  const auto peak = 20.0 * std::exp(-d2 / (2.0 * 200e3 * 200e3));
  return 1e-15 * (broad + peak);
}

/// The kernel of a shift of the 670 km discontinuity: a pattern of two
/// signs and a peak beneath the receiver.
double ShapeKernel(double theta, double phi) {
  const auto delta = FromReceiver(theta, phi);
  const auto c = std::cos(delta);
  return 1e-9 * (0.5 * (3.0 * c * c - 1.0) + 5.0 * std::exp(-delta * delta / (2.0 * 0.1 * 0.1)));
}

/// A density for the reference, of the radius alone: the core's and the
/// mantle's, each its own layers' source, so that the core-mantle boundary,
/// a node of both, takes each side's value.
double CoreDensity(double r) { return 12000.0 - 2000.0 * (r / cmb) * (r / cmb); }
double MantleDensity(double r) { return 5500.0 - 2200.0 * (r - cmb) / (surface - cmb); }

}  // namespace

int main(int argc, char** argv) {
  namespace pm = planetmodel;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " kernel.nc\n";
    return 2;
  }
  try {
    auto file = pm::ModelFile(pm::Skeleton({0.0, cmb, d670, surface}));
    file.SetName("synthetic kernel");
    file.SetLayerName(0, "core");
    file.SetLayerName(1, "lower mantle");
    file.SetLayerName(2, "upper mantle");
    file.SetInterfaceName(0, "cmb");
    file.SetInterfaceName(1, "d670");
    file.SetInterfaceName(2, "surface");
    file.SetMeta("code", "planetmodel_write_kernel");

    // the core on a coarse grid, its fields of the radius alone; the mantle
    // at the code's nodes, every 50 km, on a Gauss-Legendre grid
    const auto& skeleton = file.GetSkeleton();
    file.AddPart(0, 0, pm::ElementEdges(skeleton, 0, 0, 500e3), 2, pm::GaussLegendre(4));
    file.AddPart(1, 2, pm::ElementEdges(skeleton, 1, 2, 50e3), 2, pm::GaussLegendre(64));

    file.AddField("rho", {{0, pm::LayerSource::Radial(CoreDensity)},
                          {1, pm::LayerSource::Radial(MantleDensity)},
                          {2, pm::LayerSource::Radial(MantleDensity)}});

    // a kernel per unit volume of a dimensionless objective: s m^-4
    const auto shear = pm::FieldSpec{
        .character = pm::SCALAR,
        .dimensions = pm::Dimensions{.length = -4, .time = 1},
        .meaning = "derivative of the objective with respect to vs, per unit volume"};
    file.AddField("K_vs", shear,
                  {{1, pm::LayerSource::Lateral(ShearKernel)},
                   {2, pm::LayerSource::Lateral(ShearKernel)}},
                  {.lateral = pm::LateralRule::Linear});

    // a kernel per unit area of a shift of the interface: m^-3
    const auto shape = pm::FieldSpec{
        .character = pm::SCALAR,
        .dimensions = pm::Dimensions{.length = -3},
        .meaning = "derivative of the objective with respect to a shift of d670, per "
                   "unit area"};
    file.SetSurfaceGrid(pm::GaussLegendre(64));
    file.AddSurfaceField(
        "K_d670", shape,
        {{1, pm::LayerSource::Lateral([](double, double theta, double phi) {
            return ShapeKernel(theta, phi);
          })}},
        pm::LateralRule::Linear);

    pm::WriteModelFile(argv[1], file);
    std::cout << "wrote " << argv[1] << "\n";
  } catch (const pm::Error& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
