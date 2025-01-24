import spartann as sa
import argparse


# Text Variables
HELP_RASTER_FILE = "A raster file containing data organized in bands, with the same order as used during training."
HELP_MODELS_FILE = "Path and filename for the trained models to be used for prediction."
HELP_OUT_FILE = "Path and filename for the raster with predictions."
HELP_BLOCKSIZE = "Block size for dividing raster processing (default = 250 units)."
HELP_NCORES = "How many CPU core to use for predicting."
TITLE = "SpartANN - Spectral Pattern Analysis and Remote-sensing Tool with Artificial Neural Networks"
CITATION = "Citation will be soon available."


parser = argparse.ArgumentParser(description=TITLE, epilog=CITATION)
parser.add_argument("raster_file", nargs="?", help=HELP_RASTER_FILE)
parser.add_argument("models_file", nargs="?", help=HELP_MODELS_FILE)
parser.add_argument("out_file", nargs="?", help=HELP_OUT_FILE)
parser.add_argument("-s", "--blocksize", type=int, default=250, help=HELP_BLOCKSIZE)
parser.add_argument("-c", "--ncores", type=int, default=1, help=HELP_NCORES)
args = parser.parse_args()

if __name__ == "__main__":
    rst = sa.Raster.from_file(args.raster_file)
    ap = sa.AnnPredict.from_modelsfile(args.models_file)
    pred = ap.predictFromRaster(rst, blocksize=args.blocksize,
       ncores=args.ncores)
    pred.writeRaster(args.out_file)
