"""
Compare the fluxes in pairs of fields that have overlapping apertures
"""

import mosaic
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

def compare(lineid, specid, column='flux'):
    fieldpairs = mosaic.find_overlaps(posid="positions_mosaic")
    fitsfilename = "%s_mosaic_%s.fits" % (lineid, specid)
    table = mosaic.read_fits_table(fitsfilename)
    for fieldpair, indices in fieldpairs.items():
        field, otherfield = fieldpair
        values = table[column][[i-1 for i in indices.keys()]]
        otherindices = list()
        seps = list()
        for neighbours in indices.values():
            # first attempt - just use the nearest neighbour
            sep, otherindex = neighbours[0]
            otherindices.append(otherindex)
            seps.append(sep)
        othervalues = table[column][[i-1 for i in otherindices]]
        seps = np.array(seps)

        # plot the data
        try:
            datamax = 1.1*max(values.max(), othervalues.max())
        except ValueError:
            # something wrong with this pair: skip it
            continue
        plt.plot(values, othervalues, "*")
        plt.axis([0.0, datamax, 0.0, datamax])
        # plt.axis('scaled')
        plt.xlabel(field)
        plt.ylabel(otherfield)

        # plot line y =x
        x = np.linspace(0.0, datamax)
        plt.plot(x, x, 'r--')

        # Median gradient
        medgrad = np.median(othervalues/values)
        y = medgrad*x
        plt.plot(x, y, 'g')

        # Add info to title
        plt.title("%s %s %s, N = %i, median(%s/%s) = %.2f, sep = %.2f +/- %.2f" 
                  % (lineid, specid, column, len(values),
                     otherfield, field, medgrad,
                     seps.mean(), seps.std()),
                  fontsize="small")

        plt.savefig("%s-%s-compare-%s-%s-%s.png" 
                    % (lineid, specid, column, field, otherfield))
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare overlapping fields from Manuel's PPAK data")
    parser.add_argument("lineid", type=str, help="Emission line label")
    parser.add_argument("specid", type=str, help="Spectral range label")
    parser.add_argument("--datacolumn", type=str, default="flux",
                        help="Data column to use from mosaic file")
    cmdargs = parser.parse_args()
    compare(cmdargs.lineid, cmdargs.specid, cmdargs.datacolumn)


