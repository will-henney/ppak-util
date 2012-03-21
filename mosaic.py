"""
Utilities for working with PPAK mosaic datasets from Manuel

Initial version: WJH 14 Mar 2012
"""

import pyfits
import numpy as np
import warnings

VERBOSE = False

def read_positions(fname="positions_mosaic.dat"):
    """
    Read the mosaic fiber data from the ascii file that Manuel provided

    Returns a np.recarray with the following fields:

    'index' : line number
    'ap'    : number of aperture
    'x',    : x position of aperture [???UNITS???]
    'y',    : y position of aperture [???UNITS???]
    'Field' : which field the aperture is from
              (capital letter since there is already a recarray.field() method) 
    'flag'  : ???CHECK???
    """
    dtype = {
        'names' : ('index', 'ap', 'x', 'y', 'Field', 'flag'),
        'formats' : ('i4', 'i4', 'f4', 'f4', 'S7', 'i4')
        }
    try: 
        posdata = np.loadtxt(fname, dtype=dtype).view(type=np.recarray)
    except ValueError:          # maybe the file has a header row - try and skip it
        posdata = np.loadtxt(fname, dtype=dtype, skiprows=1).view(type=np.recarray)

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

def rescale_fluxes_per_field(fluxes, fields, datafile="overlap_factors.dat"):
    """
    Rescale all the fluxes in each field by a constant value

    Example usage: 

             table.flux = rescale_fluxes_per_field(table.flux, pos["field"])
    """
    # make a dict of rescale factors for each field
    try: 
        factordict = dict(np.loadtxt(datafile, dtype="S8, f"))
        for field, factor in factordict.items():
            fluxes[fields==field] /= factor 
    except IOError:
        warnings.warn("Flux rescale failed: error reading from %s" % (datafile))
    return fluxes


def interpolate_image(x, y, values, bbox=None, delta=None, method='nearest', full=True):
    """
    Interpolate an image from sequences of positions and values

    Input: 

    x      : sequence of x coordinates
    y      : sequence of y coordinates
    values : sequence of values (x, y, and values should be all same size)

    bbox   : bounding box of image [xmin, xmax, ymin, ymax] (default: fit to data)
    delta  : size of pixels in same units as x, y (default: deduce from data)
    method : interpolation method (default: "nearest" using scipy.interpolate.griddata)
    full   : whether to return WCS info dict as well as image (default False)

    Output: 

    image   : interpolated image
    wcsdict : (optional) dict of WCS info (CRPIX1, CRVAL1, etc.) 
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

    # Calculate the WCS info for the fits file
    # See Sec 8 of x-bdsk://Pence:2010 
    # http://adsabs.harvard.edu/abs/2010A%26A...524A..42P
    wcsdict = dict(
        CRVAL1 = xmin, CRVAL2 = ymin, 
        CRPIX1 = 1, CRPIX2 = 1,
        PC1_1 = 1.0, PC1_2 = 0.0, PC2_1 = 0.0, PC2_2 = 1.0,
        CDELT1 = dx, CDELT2 = dy,
        CTYPE1 = "PIXEL", CTYPE2 = "PIXEL"
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

    if full:
        return im, wcsdict
    else:
        return im
    
    
def write_fits_images(emline, specid="sb", delta=0.5, 
                      method="nearest", rescale=False, pos=None):
    """
    Write a series of interpolated FITS images for a single emission line

    Write one image file for each data column in the mosaic
    table. Image file names are of the form:

    EMLINE-SPECID-VAR-METHOD-DELTA.fits

    EMLINE is the emission line label
    SPECID is the spectral range label (e.g., "sb" for short blue, etc)
    VAR is the data column variable that is being mapped (e.g., flux, center, etc)
    METHOD is the interpolation method
    DELTA is the pixel size in units of 0.1 arcsec
    
    Example: FeIII4658-sb-flux-nearest-05.fits
    """
    if pos is None:
        pos = read_positions()
    table = read_fits_table("%s_mosaic_%s.fits" % (emline, specid))
    
    if rescale:                 # Optionally perform rescaling
        table.flux = rescale_fluxes_per_field(table.flux, pos.Field)

    for name in table.names:
        hdu = pyfits.PrimaryHDU()
        im, wcsdict = interpolate_image(pos.x, pos.y, table[name], 
                                        delta=delta, method=method, full=True)
        hdu.data = im[:,::-1]   # why do we need to reflect the x-axis?
        hdu.header.update("LINE", emline)
        hdu.header.update("SPECID", specid)
        hdu.header.update("VARIABLE", name)
        hdu.header.update("INTERP", method)
        for k, v in wcsdict.items():
            hdu.header.update(k, v)
        outfilename = "%s-%s-%s-%s-%2.2i.fits" % (
            emline, specid, name, method, 10*delta)
        if VERBOSE:
            print "Image shape: ", im.shape
            print "Writing FITS file: ", outfilename
        hdu.writeto(outfilename, clobber=True)
    return True


def write_fits_ratios(ratio, specrange="green", delta=0.5, method="nearest"):
    """
    Similar to write_fits_images() but for the ratio files

    ratio : e.g., dSII, dClIII, dOII
    specrange : either "red", "green", or "blue"
    """
    pos = read_positions("positions_%s_long.dat" % (specrange))
    table = read_fits_table("M42_%s.fits" % (ratio))

    for name in table.names:
        hdu = pyfits.PrimaryHDU()
        mask = ~np.isnan(table[name])
        im, wcsdict = interpolate_image(pos.x[mask], pos.y[mask], table[name][mask], 
                                        delta=delta, method=method, full=True)
        hdu.data = im[:,::-1]   # why do we need to reflect the x-axis?
        hdu.header.update("RATIO", ratio)
        hdu.header.update("SPECRANG", specrange)
        hdu.header.update("VARIABLE", name)
        hdu.header.update("INTERP", method)
        for k, v in wcsdict.items():
            hdu.header.update(k, v)
        outfilename = "%s-%s-%s-%s-%2.2i.fits" % (
            ratio, specrange, name, method, 10*delta)
        if VERBOSE:
            print "Image shape: ", im.shape
            print "Writing FITS file: ", outfilename
        hdu.writeto(outfilename, clobber=True)
    return True

def process_all_lines(method="nearest", delta=0.5, rescale=True):
    """
    Call write_fits_images for each emission line mosaic in current directory
    """
    import glob
    fallback_posfile = "positions_mosaic.dat"
    mosaicfiles = glob.glob("*_mosaic_*.fits")
    for mosaicfile in mosaicfiles:
        emline, middle, specid = mosaicfile.split(".")[0].split("_")

        # New version of position files 21 Mar 2012
        # Separate file for long and short exposures
        if specid.startswith("s"):
            posfile = "positions_mosaic_long.dat"
        elif specid.startswith("l"):
            posfile = "positions_mosaic_short.dat"
        else:
            warnings.warn("Unrecognised spectral range: %s" % (specid))
            posfile = fallback_posfile

        # Try to cover all possibilities for where the position file might be
        try:
            pos = read_positions(posfile)
        except IOError:
            try: 
                pos = read_positions("../" + posfile)
            except IOError:
                pos = read_positions(fallback_posfile)

        if VERBOSE:
            print "Writing FITS files for ", emline, specid, delta, method
        write_fits_images(emline, specid=specid, delta=delta, 
                          method=method, rescale=rescale, pos=pos)
    return True

def find_overlaps_slow(diameter=4.0, posid="positions_green_long", nmax=None):
    """
    Find apertures that overlap between any two adjacent fields

    The aperture size is 2.68 arcsec, but we take a slightly larger diameter by default

    The algorithm is rather inefficient and counts every pair twice.

    nmax : If set, this is maximum number of apertures to process - only for testing
    """

    pos = read_positions(posid + ".dat")

    fieldset = set(pos.Field)

    if nmax is None:
        nmax = len(pos)
    # simple loop algorithm
    pairdict = dict()           # dict to save the pairs of overlaps
    for i in range(nmax):
        thisField = pos.Field[i]
        thisindex = pos.index[i]
        x0, y0 = pos.x[i], pos.y[i]
        others = pos[pos.Field != thisField]
        distances = np.apply_along_axis(np.linalg.norm, 0, (others.x-x0, others.y-y0))
        overlapmask = distances <= diameter
        otherindices = others.index[overlapmask]
        if len(otherindices) > 0:
            otherFields = others.Field[overlapmask]
            otherseps = distances[overlapmask]
            # print thisField, pos.index[i]
            for otherindex, otherField, othersep in zip(otherindices, otherFields, otherseps):
                # print "--> ", otherField, otherindex, othersep
                pairdict[(int(thisindex), int(otherindex))] = dict(
                    Fields=(thisField, otherField), sep=float(othersep)
                    )

    return pairdict

def find_overlaps(diameter=2.68, maxsep=3.5, posid="positions_green_long"):
    """
    Find apertures that overlap between any two adjacent fields

    The aperture diameter is 2.68 arcsec, which is the maximum
    separation that will produce overlap.  If there is at least one
    overlap, then we also look for other neighbours at slightly larger
    separations up to maxsep.

    The algorithm is much much faster the previous one (factor of 50
    at least), but can potentially use a lot of memory.
    """

    pos = read_positions(posid + ".dat")

    fieldset = set(pos.Field)

    # Make 2d array of separations between pairs
    S = np.sqrt((pos.x - pos.x[:,np.newaxis])**2 + (pos.y - pos.y[:,np.newaxis])**2)

    # Create a mask that contains only elements in the upper triangle above diagonal
    # This is so that we eliminate self-separations (diagonal) and double counting
    mask = np.zeros_like(S, dtype=np.bool)
    mask[np.triu_indices_from(S)] = True
    mask[np.diag_indices_from(S)] = False
    # Now restrict mask to pairs with small enough separation
    mask = mask & (S < maxsep)
    # convert mask to lists of indices
    I, J = np.nonzero(mask)
    # and zip them into a list of pairs of indices
    indexpairs = zip(I, J)

    #
    # Now pack all the results into a nicely organized data structure
    #

    # Make some convenience arrays with only the masked points
    Iindices = pos.index[I]
    Jindices = pos.index[J]
    IFields = pos.Field[I]
    JFields = pos.Field[J]
    seps = S[mask]
    # Note that we don't actually use the ap values anywhere

    # The return value of this function is pairdict, which is a dict
    # with keys that are 2-tuples of fields that overlap.  The values
    # are dicts keyed by the indices of the apertures in the first
    # field.  The values in /these/ dicts are lists of (sep, index)
    # pairs, where the index is of the neighbouring aperture in the
    # second field and the list is ordered by separation (closest
    # first).
    # 
    # Example of part of one item of pairdict is:
    #         { ...., 
    #           ('c7_lg', 'c19_lg'):  {4157: [(1.2907702, 7097),
    #                                         (2.1837666, 7042),
    #                                         (2.8555672, 7021),
    #                                         (3.1828291, 7057)],
    #                                  4158: [(1.2907702, 7089),
    #                                         (2.1736367, 7054),
    #                                         (2.8703377, 7024),
    #                                         (3.1819005, 7074)],
    #                                  .... }
    #            .... }
    #
    fieldpairset = set(zip(pos.Field[I], pos.Field[J]))
    pairdict = dict()
    for fieldpair in fieldpairset:
        thisIField, thisJField = fieldpair
        if thisIField == thisJField:
            continue            # skip neighbours in the same field
        thisfieldpairmask = (IFields==thisIField) & (JFields==thisJField)
        indexset = set(Iindices[thisfieldpairmask])
        indexdict = dict()
        for thisIindex in indexset:
            neighbourmask = (thisfieldpairmask) & (Iindices==thisIindex)
            neighbour_seps = seps[neighbourmask]
            if neighbour_seps.min() > diameter:
                continue        # skip cases where there is no actual overlap
            neighbour_indices = Jindices[neighbourmask]
            indexdict[thisIindex] = sorted(zip(neighbour_seps, neighbour_indices))
        pairdict[fieldpair] = indexdict

    return pairdict



def fieldpair_sortkey(pair):
    "Rewrite the field id pairs so that lexical sorting works properly"
    def leftpadzero(s):
        "Changes 'c1_lg_1' -> 'c001_lg_1', etc"
        parts = s.split("_")
        parts[0] = parts[0][0] + "%3.3i" % (int(parts[0][1:]))
        return "_".join(parts)
    return tuple([leftpadzero(s) for s in pair])

