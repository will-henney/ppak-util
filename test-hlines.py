import mosaic
import pyfits

pos = mosaic.read_positions()
tables = dict(
    Hgamma = mosaic.read_fits_table("Hgamma_mosaic_sb.fits"),
    Hdelta = mosaic.read_fits_table("Hdelta_mosaic_sb.fits")
    )

# Write a FITS file for each data column of each emission line
for emline, table in tables.items():
    for name in table.names:
        im = mosaic.interpolate_image(pos.x, pos.y, table[name], delta=0.5)
        pyfits.writeto(
            "%s-%s.fits" % (emline, name), 
            im[:,::-1], 
            clobber=True)

# And do the flux ratio of the two lines
ratio = tables["Hdelta"].flux / tables["Hgamma"].flux
im = mosaic.interpolate_image(pos.x, pos.y, ratio, delta=0.5)
pyfits.writeto("Hd-Hg-ratio.fits", im[:,::-1], clobber=True)
