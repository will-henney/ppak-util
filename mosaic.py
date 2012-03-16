"""
Utilities for working with PPAK mosaic datasets from Manuel

Initial version: WJH 14 Mar 2012
"""

import pyfits
import numpy as np

VERBOSE = False

def read_positions(fname="positions_mosaic.dat"):
    """
    Read the mosaic fiber data from the ascii file that Manuel provided

    Returns a np.recarray with the following fields:

    'index' : line number
    'ap'    : number of aperture
    'x',    : x position of aperture [???UNITS???]
    'y',    : y position of aperture [???UNITS???]
    'field' : which field the aperture is from
    'flag'  : ???CHECK???
    """
    dtype = {
        'names' : ('index', 'ap', 'x', 'y', 'field', 'flag'),
        'formats' : ('i4', 'i4', 'f4', 'f4', 'S5', 'i4')
        }
    return np.loadtxt(fname, dtype=dtype).view(type=np.recarray)

def read_fits_table(fname="Hgamma_mosaic_sb.fits"):
    """
    Read a FITSTABLE of the mosaic in a particular line

    The table has the following columns:
    
    'ap'           : number of aperture (fiber)
    'center',      : central wavelength from Gaussian fit [Angstrom]
    'cont',        : continuum flux level interpolated under line [erg/s/cm2/A]
    'flux',        : integrated flux of line [erg/s/cm2]
    'eqw',         : equivalent width of line (-'flux'/'cont') [A]
    'core',        : peak flux at line center [erg/s/cm2/A]
    'gfwhm'        : FWHM of Gaussian fit [A]
    """
    with pyfits.open(fname) as f:
        table = f[1].data
    return table


def interpolate_image(x, y, values, bbox=None, delta=None, method='nearest'):
    """
    Interpolate an image from sequences of positions and values

    Input: 

    x      : sequence of x coordinates
    y      : sequence of y coordinates
    values : sequence of values (x, y, and values should be all same size)

    bbox   : bounding box of image [xmin, xmax, ymin, ymax] (default: fit to data)
    delta  : size of pixels in same units as x, y (default: deduce from data)
    method : interpolation method (default: "nearest" using scipy.interpolate.griddata)

    Output: 

    image : interpolated image
    
    """
    from scipy.interpolate import griddata, Rbf

    # Determine bounding box for grid
    try:
        xmin, xmax, ymin, ymax = bbox
    except (TypeError, ValueError):
        # no sensible bbox given, so fit one to the data
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

    # Determine pixel size for grid
    if delta is None:
        # If delta not specified, find minimum distance between adjacent pixels
        delta = np.apply_along_axis(
            np.linalg.norm, 1, 
            np.diff(zip(x, y), axis=0)
            ).min()
    dx, dy = delta, delta

    # Set up grid 
    grid_x, grid_y = np.meshgrid( 
        np.arange(xmin, xmax+dx, dx), 
        np.arange(ymin, ymax+dy, dy)
        ) 

    if method in ["nearest", "linear", "cubic"]:
        # Standard interpolations
        im = griddata((x, y), values, (grid_x, grid_y), method=method)
    elif method.startswith("rbf"):
        # Interpolate with Radial Basis Functions
        # http://en.wikipedia.org/wiki/Radial_Basis_Function
        # This is still "experimental" as of 14 Mar 2012 (that is, it doesn't work)
        rbfmethod = method.split()
        if len(rbfmethod) == 1:
            function = "multiquadric"
        else:
            function = rbfmethod[1]
        rbf = Rbf(x, y, values, function=function)
        im = rbf(grid_x, grid_y)

    return im
    
    
def write_fits_images(emline, specid="sb", delta=0.5, method="nearest"):
    """
    Write a series of interpolated FITS images for a single emission line

    Write one image file for each data column in the mosaic
    table. Image file names are of the form:

    EMLINE-VAR-METHOD-DELTA.fits

    EMLINE is the emission line label
    VAR is the data column variable that is being mapped (e.g., flux, center, etc)
    METHOD is the interpolation method
    DELTA is the pixel size in units of 0.1 arcsec
    
    Example: FeIII4658-flux-nearest-05.fits
    """
    pos = read_positions()
    table = read_fits_table("%s_mosaic_%s.fits" % (emline, specid))
    for name in table.names:
        im = interpolate_image(pos.x, pos.y, table[name], delta=delta, method=method)
        outfilename = "%s-%s-%s-%2.2i.fits" % (emline, name, method, 10*delta)
        if VERBOSE:
            print "Image shape: ", im.shape
            print "Writing FITS file: ", outfilename
        pyfits.writeto(outfilename, im[:,::-1], clobber=True)
    return True

def process_all_lines(method="nearest", delta=0.5):
    """
    Call write_fits_images for each emission line mosaic in current directory
    """
    import glob
    mosaicfiles = glob.glob("*_mosaic_*.fits")
    for mosaicfile in mosaicfiles:
        emline, middle, specid = mosaicfile.split(".")[0].split("_")
        if VERBOSE:
            print "Writing FITS files for ", emline, specid, delta, method
        write_fits_images(emline, specid=specid, delta=delta, method=method)
    return True

def find_overlaps(diameter=2.68, pos=None):
    """
    Find apertures that overlap between any two adjacent fields
    """

    if pos is None:             # read pos data if not passed in
        pos = read_positions()

    # simple loop algorithm
    for i in range(len(pos)):
        x0, y0 = pos.x[i], pos.y[i]
        others = pos[pos.index != i+1]
        distances = np.apply_along_axis(np.linalg.norm, 0, (others.x-x0, others.y-y0))
        overlaps = others[distances < diameter]
        if len(overlaps) > 0:
            print pos.index[i], overlaps.index
            # print "*** %s ***" % (pos[i])
            # print overlaps
