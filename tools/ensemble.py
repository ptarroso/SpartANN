import spartann as sa
import numpy as np
import argparse


# Text Variables
HELP_RASTER_FILE = "A raster file containing data organized in bands to aggregate."
HELP_BANDS = "A list of bands to aggregate separated by coma (e.g. 1,2,3 to aggregate the first 3 bands. If not given, all bands are aggregates"
HELP_FUN = "A list of functions (median, mean, std, min or max) to aggregate the bands separeted by comma (e.g. mean,std for a two band raster raster with bands aggregated with mean and standard deviation - the default)"
HELP_OUT_FILE = "Path and filename for the raster with data aggregated."
HELP_BLOCKSIZE = "Block size for dividing raster processing (default = 250 units)."
HELP_NCORES = "How many CPU core to use for predicting blocks. Usefull for large raster with large number of bands to aggregate."
TITLE = "SpartANN - Spectral Pattern Analysis and Remote-sensing Tool with Artificial Neural Networks"
CITATION = "Citation will be soon available."

parser = argparse.ArgumentParser(description=TITLE, epilog=CITATION)
parser.add_argument("raster_file", nargs="?", help=HELP_RASTER_FILE)
parser.add_argument("out_file", nargs="?", help=HELP_OUT_FILE)
parser.add_argument("-b", "--bands", type=str, default="", help=HELP_BANDS)
parser.add_argument("-f", "--functions", type=str, default="mean,std", help=HELP_BANDS)
parser.add_argument("-s", "--blocksize", type=int, default=250, help=HELP_BLOCKSIZE)
parser.add_argument("-c", "--ncores", type=int, default=1, help=HELP_NCORES)
args = parser.parse_args()

def getFunctions(text):
    available = ["median", "mean", "std", "min", "max"]
    text = text.strip().lower().split(",")
    functions = {}
    for fun in text:
        if fun in available:
            functions[fun] = getattr(np, fun)
        else:
            raise ValueError(f"Use one of available functions {available}.")
    return functions

def getBands(text):
    text = text.strip().split(",")
    if text[0] == "":
        return None
    return [int(x) for x in text]

if __name__ == "__main__":
    try:
        rst = sa.Raster.from_file(args.raster_file)
        functions = getFunctions(args.functions)

        bands = getBands(args.bands)
        ncores = args.ncores

        out_rst = None
        for i, (name, fun) in enumerate(functions.items()):
            print(f"Aggregating by {name}")
            if out_rst is None:
                out_rst = rst.aggregate_bands(bands, fun=fun, ncores=ncores)
            else:
                temp = rst.aggregate_bands(bands, fun=fun, ncores=ncores)
                out_rst.addNewBand(temp.get_array(1))
            out_rst.addDescription(name, i+1)

        out_rst.writeRaster(args.out_file)

    except Exception as e:
        print(f"Error: {e}\n")
        parser.print_help()
