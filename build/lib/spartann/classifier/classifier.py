from osgeo import gdal_array
from typing import Callable
from time import ctime, sleep
from multiprocessing import Process, Queue
from spartann.datatable.datatable import DataTable
from spartann.engine.nnEngine import NN
from spartann.spatial import progressbar, Raster
from .validation import CohensKappa_Validation
from .model import Model, ModelContainer
from spartann.version import __version__
import numpy as np
from queue import Empty

class AnnClassifier:
    """The AnnClassifier provides methods for training a Artificial Neural Network with different parameters and optimizers for supervised classification."""

    def __init__(
        self,
        modelcontainer: ModelContainer,
        validation: Callable = CohensKappa_Validation,
    ):
        """Initiallise the classifier with a model container

        Args:
            modelcontainer: an initialised ModelContainer with model parameters.
            validation: an instance of Validation with a proper validation metric (default:Cohen's Kappa)

        """
        self.container = modelcontainer
        self.validation = validation
        self._data = None

    @classmethod
    def from_options(
        cls,
        inputs: list,
        outputs: list = ["Output1"],
        repetitions: int = 20,
        testpercent: int = 20,
        hl_schemes: list | list[list] = [8, 4],
        LR: float = 0.01,
        momentum: float = 0.01,
        optim: str = "Adam",
        validation: Callable = CohensKappa_Validation,
    ):
        """Initiallise the classifier with options with defaults.

        Args:
            n_inputs: Number of inputs neurons.
            n_outputs: Number of ouput neurons (default is 1)
            repetitions: Number of repetitions performed per network scheme.
            testpercent: percentage of points reserved for independent testing the training of the network (overfitting control).
            hl_schemes: list of hidden layer network schemes to test. A network with 10 inputs and 2 hidden layers with 5 and 3 neurons, plus a single output layer is simply given as [5,3].
            LR: Learning rate. Depending on the optimizer, might have different importance.
            momentum: Decay factor for the optimizer. Note: the "Adam" optimizer need two decay factors given in a list [B1, B2] (usually [0.9, 0.999]). Other optimization methods need only a float.
            optim: Optimization method. Available methods are "SGD" (stochastic gradient descend), "SimpleMomentum", "Momentum", "Adagrad", "RMSProp" and "Adam".
            validation: Validation metric to be used, depending on the data. An instance of Validatio class.
        """
        if not isinstance(hl_schemes[0], list):
            hl_schemes = [hl_schemes]
        container = ModelContainer(
            inputs,
            outputs,
            hl_schemes,
            LR,
            momentum,
            repetitions,
            optim,
            testpercent,
            str(validation),
        )
        return cls(container, validation)

    @classmethod
    def from_datatable(
        cls,
        dt: DataTable,
        repetitions: int = 20,
        testpercent: int = 20,
        hl_schemes: list | list[list] = [8, 4],
        LR: float = 0.01,
        momentum: float = 0.01,
        optim: str = "Adam",
        validation: Callable = CohensKappa_Validation,
    ):
        """Initiallise the classifier with a DataTable.

        Args:
            dt: A DataTable instance with data to train model.
            repetitions: Number of repetitions performed per network scheme.
            testpercent: percentage of points reserved for independent testing the training of the network (overfitting control).
            hl_schemes: list of hidden layer network schemes to test. A network with 10 inputs and 2 hidden layers with 5 and 3 neurons, plus a single output layer is simply given as [5,3].
            LR: Learning rate. Depending on the optimizer, might have different importance.
            momentum: Decay factor for the optimizer. Note: the "Adam" optimizer need two decay factors given in a list [B1, B2] (usually [0.9, 0.999]). Other optimization methods need only a float.
            optim: Optimization method. Available methods are "SGD" (stochastic gradient descend), "SimpleMomentum", "Momentum", "Adagrad", "RMSProp" and "Adam".
            validation: Validation metric to be used, depending on the data. An instance of Validatio class.
        """
        if not isinstance(hl_schemes[0], list):
            hl_schemes = [hl_schemes]
        container = ModelContainer(
            dt.datacolnames,
            dt.class_names,
            hl_schemes,
            LR,
            momentum,
            repetitions,
            optim,
            testpercent,
            str(validation),
        )
        anncl = cls(container, validation)
        anncl._data = dt
        return anncl

    def __len__(self):
        """Returns the length of stored best networks of the classifier."""
        return len(self.container)

    def __str__(self):
        """String representation providing a summary of the classifier parameters and trained networks."""
        return str(self.container)

    @property
    def getmodels(self):
        """Returns a ModelContainer with trained models."""
        return self.container

    def trainModel(
        self,
        datatable: DataTable|None = None,
        batch_size: int = 1,
        burnin = 250,
        min_train = 0,
        min_test = 0,
        maxiter: int = 10000,
        stable: int = 250,
        stable_val: float = 0.001,
    ):
        """Trains a model ensemble using an artificial neural network (ANN)

        This function trains an ANN using data and classification labels provided in a DataTable. The training process follows user-defined parameters, including batch size, stopping criteria, and minimum performance thresholds.

        Args:
            datatable (DataTable): A dataset containing labeled points for supervised learning.
            batch_size (int, optional): The number of samples per training batch. Must be between 1 and the total number of available samples. The default is 1, meaning weight updates occur after every sample. Larger batch sizes may provide smoother training.
            burnin (int, optional): The number of initial iterations to exclude. This allows the network to adjust its weights before selecting valid models, reducing the likelihood of retaining near-random networks.
            min_train (float, optional): The minimum training accuracy required to accept a network. If this threshold is not met, training is repeated.
            min_test (float, optional): The minimum testing accuracy required to accept a network. If this threshold is not met, training is repeated.
            maxiter (int, optional): The maximum number of training iterations.
            stable (int, optional): The number of consecutive iterations with an error difference below `stable_val` required to trigger early stopping.
            stable_val (float, optional): The threshold for error difference between iterations to be considered stable, used for early stopping.
        """
        if datatable:
            if isinstance(datatable, DataTable):
                self._data = datatable
            else:
                raise TypeError("Provided data must be an instance of DataTable.")
        else:
            if not self._data:
                raise ValueError("Provide data in a datatable to train the models.")

        targets = self._data.getClasses
        patterns = self._data.getData

        n_input = patterns.shape[1]
        n_output = targets.shape[1]

        if n_input != self.container.n_inputs or n_output != self.container.n_outputs:
            raise ValueError(
                f"Data must have {self.container.n_inputs} inputs and {self.container.n_outputs} targets."
            )

        net_counter = 0

        for hl_scheme in self.container.hl_schemes:
            print(
                f"Training networks with scheme i:[{n_input}] | hl:{hl_scheme} | o:[{n_output}]."
            )
            scheme = [n_input] + hl_scheme + [n_output]

            n_test = int(len(self._data) * self.container.testpercent / 100.0)
            n_train = len(self._data) - n_test

            mask = np.array([True] * n_train + [False] * n_test)

            rep = 0
            while rep < self.container.repetitions:
                print(f"Repetition: {rep+1} from {self.container.repetitions}")

                # Prepare train and test data
                np.random.shuffle(mask)
                tgt_train = targets[mask].reshape(n_train, n_output).tolist()
                tgt_test = targets[np.invert(mask)].reshape(n_test, n_output).tolist()
                pat_train = patterns[mask,].tolist()
                pat_test = patterns[np.invert(mask),].tolist()

                # Start a ANN based on provided scheme
                nn = NN(
                    scheme,
                    iterations=1,
                    LR=self.container.lr,
                    momentum=self.container.momentum,
                    optim=self.container.optimizer,
                )

                scale = self._data.is_scaled

                if scale:
                    nn.means = self._data.scale_means.tolist()
                    nn.sdevs = self._data.scale_sdevs.tolist()

                nn.initWeights()

                best = [float('-inf'), 0, ""]
                tracker = []
                err_dif = -1

                for i in range(burnin):
                    nn.trainnet(pat_train, tgt_train, batch_size = batch_size, scale=not scale, verbose=0)
                    print(f"Burn-in iteration: {i+1}", end="\r")
                print("")

                print("| Iteration |   Error   |  Train |  Test  |  Product |  ErrDiff |")

                for i in range(maxiter):
                    nn.trainnet(pat_train, tgt_train, batch_size = batch_size, scale=not scale, verbose=0)
                    pred_train = nn.testnet(pat_train, scale=not scale, verbose=0)
                    pred_test = nn.testnet(pat_test, scale=not scale, verbose=0)
                    k_train = self.validation.calc(tgt_train, pred_train)
                    k_test = self.validation.calc(tgt_test, pred_test)
                    k_prod = k_train * k_test
                    err = nn.netTrainError[0]

                    if (k_train >= min_train and k_test >= min_test
                        and k_prod > best[0]):
                        best = [k_prod, i, str(nn)]

                    if i > 1:
                        err_dif = tracker[-1][0] - err
                    tracker.append([err, k_train, k_test, k_prod, err_dif])

                    tt = sum(
                        [x[4] <= stable_val and x[4] > 0 for x in tracker[-stable:]]
                    )
                    print(
                        f"| {i:9d} | {err:9.5f} | {k_train: 5.3f} | {k_test: 5.3f} |  {k_prod: 5.3f}  | {err_dif:8.5f} |",
                        end="\r",
                        flush=True
                    )
                    if tt >= stable:
                        break

                if best[0] > float('-inf'):
                    self.container.add_model(Model(best[2], rep, scheme, tracker, best[1]))
                    print(
                        "\nBest net:"
                        + f"\n\tIteration {best[1]}"
                        + f"\n\tError: {tracker[best[1]][0]:.3f}"
                        + f"\n\tValidation train: {tracker[best[1]][1]:.3f}"
                        + f"\n\tValidation test: {tracker[best[1]][2]:.3f}"
                        + f"\n\tValidation product: {tracker[best[1]][3]:.3f}"
                    )
                    net_counter += 1
                    rep += 1
                else :
                    print("\nTraining failed minimum values (train > "
                        + f"{min_train} and test > {min_test}.\nRepeating "
                        + "network training.")

    def writeModel(self, filename: str):
        """Writes the trained models to a file.

        Args:
            filename: the file name to be written, usually with .obj extension.

        """
        self.container.save(filename)


class AnnPredict:
    """Class for predicting based on trained models."""

    def __init__(self, modelcontainer):
        """Initialise class

        Args:
            modelcontainer: an instance of ModelContainer with the best trained model with ANNClassifier.
        """
        self.container = modelcontainer

    @classmethod
    def from_annclassifier(cls, ann: AnnClassifier):
        """Open models and predict from a AnnClassifier instance with trained networks.

        Args:
            ann: instace of AnnClassifier with trained models.

        """
        if isinstance(ann, AnnClassifier):
            return cls(ann.getmodels)
        else:
            raise TypeError("Provide an instance of AnnClassifier")

    @classmethod
    def from_modelsfile(cls, filename: str):
        """Open models from file for prediction.

        Args:
            filename: the models file saved from an AnnClassifier instance. Usually .obj file.

        """
        container = ModelContainer.load(filename)
        return cls(container)

    def __str__(self):
        return str(self.container)

    def predict(self,
        patterns: list,
        scale: bool = True) -> list:
        """Predict output of model to a list of patterns.

        Args:
            patterns: a list with patterns to be tested with trained models. Patterns must follow the same order as in training.
            scale: if patterns should be scaled with network stored means and standard deviations.

        Returns:
            a list of predictions with the format [[[output_1], [output_2]]] for all models trained.

        """
        predictions = []
        for net in self.container.get_best_nets():
            nn = NN.loadnet(net)
            predictions.append(nn.testnet(patterns, scale=scale, verbose=0))
        return predictions


    def predictFromDataTable(self, datatable: DataTable, scale: bool = True) -> DataTable:
        """Predict values with data from a DataTable instance.

        Uses the data slot of a DataTable to get values to predict.

        Args:
            datatable: a DataTable instance with data to test and predict. Must follow the same order as original data used to train models.
            scale: If true, use network stored mean and standard deviations to scale data.

        """
        if datatable.datacolnames != self.container.inputs:
            print("Warning: The data column names do not match the model's input names. Predictions are still being made, but please ensure that the order of the data columns is correct.")

        patterns = datatable.getData.tolist()
        predtable = DataTable(
            datatable.getPoints.tolist(), datatable.getClasses.tolist()
        )
        pred = self.predict(patterns, scale)
        for i in range(len(pred)):
            predtable.add_datacolumns(pred[i], datatable.colnames[i])
        return predtable

    def predictFromRaster(
        self,
        raster: Raster,
        scale: bool = True,
        multiplier: int = 1000,
        blocksize: int = 250,
        nodata: int | float = -9999,
        dtype: str = "int16",
        ncores: int = 1
    ) -> Raster:
        """Predicts from a raster object used trained models.

        The bands of the raster object are used as patterns and must be provided in the same order as original data used for training.
        Multicore processing is usefull for large rasters. It requires that the
        raster exists as a file (either loaded form file or written to one).

        Args:
            raster: an instance of a Raster object.
            scale: If True, use means and standard deviation stored with the network to scale data.
            multiplier: a multiplication factor for the predictions values
            blocksize: the size of the blocks to predict (instead of using the whole raster array at once).
            nodata: nodata value to be used in the new raster
            dtype: the data type of the raster to be created
            ncores: Number of cores/processes to be used for processing a raster

        Returns:
            A Raster object with predictions, where each band represents a trained network (repetitions and schemes). Check metadata.

        """

        if raster.bandnames != self.container.inputs:
            print("Warning: The raster band names do not match the model's input names. Predictions are still being made, but please ensure that the order of the bands is correct.")

        n = len(self.container)
        n_outputs = self.container.n_outputs

        pred_rst = Raster.from_scratch(
            raster.size,
            raster.res,
            raster.origin,
            bands=n * n_outputs,
            nodata=nodata,
            projwkt=raster.proj,
            dtype=gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(dtype)),
        )

        # Set Metadata
        md = {
            "Created_with": f"SpartANN, version {__version__}",
            "Creation_date": ctime(),
            "Number_of_predictions": f"{n*n_outputs} from {n} models with {n_outputs} outputs",
            "Multiplier": 1 / multiplier,
        }
        pred_rst.addMetadata(md)

        if ncores == 1:
            self._singlecore_raster_predict(raster, pred_rst, blocksize, scale,
                                multiplier, nodata, dtype)
        else:
            self._multicore_raster_predict(raster, pred_rst, blocksize, scale,
                                multiplier, nodata, dtype, ncores)

        # Sets bands specific metadata
        for o in range(n_outputs):
            for i in range(n):
                md = {
                    "Scheme": self.container.models[i].scheme,
                    "Repetition": self.container.models[i].repetition,
                    "Output": self.container.outputs[o],
                }
                descr = f'Prediction for {self.container.outputs[o]}, with scheme {md["Scheme"]}, repetition {md["Repetition"]}'
                pred_rst.addMetadata(md, i + (n * o) + 1)
                pred_rst.addDescription(descr, i + (n * o) + 1)

        return pred_rst

    def _singlecore_raster_predict(self,
        raster: Raster,
        pred_rst: Raster,
        blocksize: int,
        scale: bool,
        multiplier: int,
        nodata: int | float,
        dtype: str):
        """
        Internal function for raster prediction using a single-core
        implementation.

        This function avoids the overhead of creating additional processes and
        directly provides predictions for the raster.
        """

        n_outputs = self.container.n_outputs
        n = len(self.container)

        progressbar(0, 1)

        r_iter = raster.block_iter(blocksize=blocksize, read_arr=True)

        for block in r_iter:
            pos, arr = block
            b, r, c = arr.shape

            ## to table format (pixels, bands)
            arr = arr.transpose(1, 2, 0).reshape(r * c, b)
            pred = np.array(self.predict(arr.tolist(), scale=scale))
            pred = pred.transpose(2, 0, 1).reshape(n * n_outputs, r, c)
            pred = pred * multiplier
            pred[np.isnan(pred)] = nodata
            pred = pred.astype(dtype)
            pred_rst.set_array(pred, rc=pos[:2])

            progressbar(pos[2], pos[3])

    def _multicore_raster_predict(self,
        raster: Raster,
        pred_rst: Raster,
        blocksize: int,
        scale: bool,
        multiplier: int,
        nodata: int | float,
        dtype: str,
        ncores: int):
        """
        Internal function for raster prediction using a multi-core
        implementation.

        This function spawns a set of workers to process the raster with
        minimal memory usage. The raster is divided into blocks for efficient
        processing. To circumvent the single-process raster access limitation,
        each process reads the raster file directly. Therefore, the raster must
        either be written to or loaded from a file.

        It utilizes task and result queues to provide data to and retrieve
        results from the worker functions. Results are processed as they become
        available in the queue, ensuring continuous and efficient data
        processing.
        """
        progressbar(0, 1)

        tasks = Queue()
        results = Queue()

        if not raster.hasSource:
            msg = "The raster must be associated with a file (either loaded " +\
                  "from or written to one) to enable multicore processing."
            raise ValueError(msg)

        file = raster.source

        r_iter = raster.block_iter(blocksize=blocksize, read_arr=False)

        for block in r_iter:
            tasks.put(block)

        processes = []

        for wid in range(ncores):
            p = Process(target=self._worker,
                        args=(tasks, scale, file, blocksize, results))
            processes.append(p)
            p.start()

        progress = []
        while any(p.is_alive() for p in processes) or not results.empty():
            try:
                # Try to get a result from the result queue (non-blocking)
                pos, pred = results.get_nowait()
                progress.append(pos[2])
                pred = pred * multiplier
                pred[np.isnan(pred)] = nodata
                pred = pred.astype(dtype)
                pred_rst.set_array(pred, rc=pos[:2])
                progressbar(max(progress), pos[3])
            except Empty:
                # To result retrieve, wait for next one
                pass
            except Exception as e:
                print(f"{type(e).__name__} - {e}")
                break
            # Allow a small sleep to avoid tight polling loop
            sleep(0.1)

        for p in processes:
            p.join()

    def _worker(self,
            tasks: Queue,
            scale: bool,
            file: str,
            blocksize: int,
            results: Queue):
        """
        Internal worker function for multi-core raster processing.

        This function predicts raster values for a specific block of data. It
        reads the required subset of data directly from the file and applies
        all trained models found in the model container. A queue-based strategy
        is used for retrieving data and sending results, ensuring minimal
        memory usage.
        """
        while True:
            try:
                pos = tasks.get_nowait()
                rst = Raster.from_file(file)
                arr = rst.get_subarray(pos[0][:2], blocksize)
                b, r, c = arr.shape
                n_outputs = self.container.n_outputs
                n = len(self.container)
                ## to table format (pixels, bands)
                arr = arr.transpose(1, 2, 0).reshape(r * c, b)
                pred = np.array(self.predict(arr.tolist(), scale=scale))
                pred = pred.transpose(2, 0, 1).reshape(n * n_outputs, r, c)
                results.put((pos[0], pred))
            except Empty:
                # No tasks to process
                break
            except Exception as e:
                # Handle unexpected exceptions
                print(f"{type(e).__name__} - {e}")
                break
