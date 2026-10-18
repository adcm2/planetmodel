// Includes the netCDF header, which refuses to compile without netCDF-C.
#include <iostream>

#include "planetmodel/netcdf.hpp"

int main() {
  std::cout << "netCDF-C " << nc_inq_libvers() << '\n';
  return 0;
}
