// Uses the parts of the library that need no netCDF-C.
#include <iostream>

#include "planetmodel/units.hpp"
#include "planetmodel/version.hpp"

int main() {
  const auto scales = planetmodel::Scales::Geophysical(6371e3);
  std::cout << "planetmodel " << planetmodel::Version() << ": G = "
            << planetmodel::G_SI / scales.Factor(planetmodel::units::GRAVITATIONAL_CONSTANT)
            << '\n';
  return 0;
}
