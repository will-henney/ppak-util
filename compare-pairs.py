"""
Compare the fluxes in pairs of fields that have overlapping apertures
"""

import mosaic
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import argparse
import asciitable

def compare(lineid, specid, column='flux', posid="positions_mosaic"):
    fieldpairs = mosaic.find_overlaps(posid=posid)
    fitsfilename = "%s_mosaic_%s.fits" % (lineid, specid)
    table = mosaic.read_fits_table(fitsfilename)

    # initialise a table to save the results
    outtable = dict(field1=list(), field2=list(), 
                    s_mean=list(), s_std=list(), N=list(), 
                    rat_med=list(), rat_q25=list(), rat_q75=list())
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
        # The color of the points represents the separations
        plt.scatter(x=values, y=othervalues, c=seps, 
                    marker="o", vmin=0.0, vmax=2.68, alpha=0.6)
        plt.axis([0.0, datamax, 0.0, datamax])
        cb = plt.colorbar()
        cb.set_label("aperture\nseparation")
        # plt.axis('scaled')
        plt.xlabel(field)
        plt.ylabel(otherfield)

        # plot line y =x
        x = np.linspace(0.0, datamax)
        plt.plot(x, x, 'r--')

        # Median gradient
        medgrad = np.median(othervalues/values)
        quart25 = scipy.stats.scoreatpercentile(othervalues/values, 25)
        quart75 = scipy.stats.scoreatpercentile(othervalues/values, 75)
        dyplus = quart75 - medgrad
        dyminus = medgrad - quart25
        y = medgrad*x
        plt.plot(x, y, 'g')

        # Add info to title
        plt.title("%s %s %s\n N = %i, median(%s/%s) = %.2f + %.2f - %.2f\n sep = %.2f +/- %.2f" 
                  % (lineid, specid, column, len(values),
                     otherfield, field, medgrad, dyplus, dyminus,
                     seps.mean(), seps.std()),
                  fontsize="small")

        plt.savefig("%s-%s-compare-%s-%s-%s.png" 
                    % (lineid, specid, column, field, otherfield))
        plt.clf()

        # save the data to the output table
        outtable["field1"].append(field)
        outtable["field2"].append(otherfield)
        outtable["s_mean"].append(seps.mean())
        outtable["s_std"].append(seps.std())
        outtable["N"].append(len(values))
        outtable["rat_med"].append(medgrad)
        outtable["rat_q25"].append(quart25)
        outtable["rat_q75"].append(quart75)

    # Write output table to file
    fmt = "%.3f"
    fmts = dict(s_mean=fmt, s_std=fmt, 
                rat_med=fmt, rat_q25=fmt, rat_q75=fmt)
    asciitable.write(outtable, 
                     "%s-%s-compare-%s.dat" %  (lineid, specid, column),
                     delimiter="\t", formats=fmts)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare overlapping fields from Manuel's PPAK data")
    parser.add_argument("lineid", type=str, help="Emission line label")
    parser.add_argument("specid", type=str, help="Spectral range label")
    parser.add_argument("--datacolumn", type=str, default="flux",
                        help="Data column to use from mosaic file")
    parser.add_argument("--posid", type=str, default="positions_mosaic",
                        help="Name of file (sans suffix) with position data")
    cmdargs = parser.parse_args()
    compare(cmdargs.lineid, cmdargs.specid, cmdargs.datacolumn, cmdargs.posid)


