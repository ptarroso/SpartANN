import spartann as sa
import argparse


# Text Variables
HELP_DATA_FILE = "The file contains coordinates, species, and classifications used for supervised training. It follows the format 'LON;LAT;CLASS' with a header. The coordinates must be in the same coordinate reference system as the provided raster."
HELP_RASTER_FILE = "The path to the raster containing the information to build the model after it's extracted for each data point. All present bands are used."
HELP_MODELS_FILE = "Path and filename for the models trained (e.g. 'model.obj')"
HELP_REPETITIONS = "The number of repetitions for training the models (default is 20)."
HELP_TESTPERCENT = "The percentage of data points reserved for testing. Note that test points are randomized for each model repetition (default=20)."
HELP_MAXITER = "Maximum number of training iterations for the model (default=10000)."
HELP_STABLEITER = "The number of iterations where error minimization falls below the chosen stability value to halt the training process (default=250)."
HELP_STABLEVAL = "Stability value (difference to previous iteration error) that indicates convergence (default=0.001)."
HELP_LRATE = "Learning rate value (default 0.01)."
HELP_MOMENTUM = "Momentum (default 0.01)."
HELP_HIDDENLYRS = "The default structure consists of 3 hidden layers connecting the input to output layers. You can specify the hidden layers by providing the number of neurons in each layer, separated by commas (e.g., '4,3' for 2 hidden layers, with the first having 4 neurons and the second having 3 neurons)."
TITLE = "SpartANN - Spectral Pattern Analysis and Remote-sensing Tool with Artificial Neural Networks"
CITATION = "Citation will be soon availble."


startGUI = False

parser = argparse.ArgumentParser(description=TITLE, epilog=CITATION)
parser.add_argument("data_file", nargs="?", help=HELP_DATA_FILE)
parser.add_argument("raster_file", nargs="?", help=HELP_RASTER_FILE)
parser.add_argument("out_file", nargs="?", help=HELP_MODELS_FILE)
parser.add_argument("-r", "--repetitions", type=int, default=20, help=HELP_REPETITIONS)
parser.add_argument("-t", "--tpercent", type=float, default=20, help=HELP_TESTPERCENT)
parser.add_argument("-mi", "--maxiter", type=int, default=10000, help=HELP_MAXITER)
parser.add_argument("-si", "--stableiter", type=int, default=250, help=HELP_STABLEITER)
parser.add_argument(
    "-sv", "--stableval", type=float, default=0.001, help=HELP_STABLEVAL
)
parser.add_argument("-l", "--lrate", type=float, default=0.01, help=HELP_LRATE)
parser.add_argument("-m", "--momentum", type=float, default=0.01, help=HELP_MOMENTUM)
parser.add_argument("-hl", "--hiddenlyrs", type=str, default="3", help=HELP_HIDDENLYRS)
args = parser.parse_args()

if __name__ == "__main__":
    rst = sa.Raster.from_file(args.raster_file)
    dt = sa.DataTable.from_file(args.data_file)
    dt.getDataFromRaster(rst)
    dt.exclude_nan()
    dt.scaleData()

    scheme = [int(x) for x in args.hiddenlyrs.split(",")]

    ann = sa.AnnClassifier.from_datatable(
        dt=dt,
        repetitions=args.repetitions,
        testpercent=args.tpercent,
        hl_schemes=scheme,
        LR=args.lrate,
        momentum=args.momentum,
    )

    ann.trainModel(
        dt, maxiter=args.maxiter, stable=args.stableiter, stable_val=args.stableval
    )
    ann.writeModel(args.out_file)
