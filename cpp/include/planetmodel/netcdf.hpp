#ifndef PLANETMODEL_NETCDF_HPP
#define PLANETMODEL_NETCDF_HPP

/// @file
/// netCDF-C, over which the model file is read and written.  It is there
/// when the library was configured with PLANETMODEL_WITH_NETCDF, the
/// default; a code built without it cannot include this header, nor any
/// that includes it.

#ifndef PLANETMODEL_WITH_NETCDF
#error "planetmodel was configured without netCDF-C, so without the model file: configure it with PLANETMODEL_WITH_NETCDF=ON"
#endif

#include <netcdf.h>

#endif  // PLANETMODEL_NETCDF_HPP
