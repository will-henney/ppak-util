"""
Read in a mosaic dataset from Manuel

Initial version: WJH 14 Mar 2012
"""

import pyfits
import numpy as np

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
    with open(fname) as f:
        posdata = np.loadtxt(fname, dtype=dtype)
    return posdata

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

    
    
