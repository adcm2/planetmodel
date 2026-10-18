"""Pictures of models: profiles, sections, maps, spectra and VTK files.

Making a picture is kept apart from drawing it: the numbers are
`planetmodel.slices`' (a sphere, a section, a profile, a spectrum),
sampled in the core for any model, and the functions here draw them on
matplotlib axes they are given, returning what they drew and leaving
the figure to the caller.  Nothing here is particular to the Earth.

`profile` and `radial_profile` draw a field against radius, radius
upward, a step at every interface (`profiles`); `section` draws a
great-circle section as a wedge of the annulus or a disc, or in the
body's deformed shape (`sections`); `sphere_map` draws a sphere slice on
cartopy's projections (`maps`, the `maps` extra); `spectrum`,
`correlation` and `correlation_total` draw spectra and correlations by
degree and radius (`spectra`); `write_vtk` writes a model as VTK files
for ParaView (`vtk`, no extra needed); `style` holds the look they
share.  matplotlib is not imported here: the axes are the caller's,
made with the `plot` extra, and cartopy only where a map is drawn.
"""
from .maps import cell_edges, sphere_map
from .profiles import profile, radial_profile
from .sections import section
from .spectra import correlation, correlation_total, spectrum
from .style import BOUNDARY_STYLE, DIVERGING, SEQUENTIAL, colour_scale
from .vtk import write_vtk

__all__ = ["profile", "radial_profile", "section", "sphere_map", "cell_edges",
           "spectrum", "correlation", "correlation_total", "write_vtk",
           "BOUNDARY_STYLE", "DIVERGING", "SEQUENTIAL", "colour_scale"]
