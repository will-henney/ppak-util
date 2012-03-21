"""
Very simple script to take ratios of two lines
"""

import pyfits
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take ratio A/B of two interpolated line maps")
    parser.add_argument("lineA", type=str, help="Emission line label A")
    parser.add_argument("lineB", type=str, help="Emission line label B")
    parser.add_argument("specid", type=str, help="Spectral range label")
    parser.add_argument("--variable", type=str, default="flux",
                        help="Variable to use")
    parser.add_argument("--interp", type=str, default="nearest",
                        help="Interpolation method")
    parser.add_argument("--pixscale", type=int, default=5,
                        help="Pixel scale in tenths of arcsec")
    cmdargs = parser.parse_args()

    suffix = "%(specid)s-%(variable)s-%(interp)s-%(pixscale)0.2i.fits" % (vars(cmdargs))

    # Take the first HDU in each file
    hduA = pyfits.open("-".join([cmdargs.lineA, suffix]))[0]
    hduB = pyfits.open("-".join([cmdargs.lineB, suffix]))[0]

    hduA.data /= hduB.data
    hduA.header.update("VARIABLE", "%s A/B" % (cmdargs.variable))
    hduA.header.update("LINE_A", cmdargs.lineA)
    hduA.header.update("LINE_B", cmdargs.lineB)

    outfile = "-".join(["ratio", cmdargs.lineA, cmdargs.lineB, suffix])
    hduA.writeto(outfile)



