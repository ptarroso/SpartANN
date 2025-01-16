from typing import Callable
from .validation import CohensKappa_Validation
from spartann.version import __version__
import pickle


class Model:
    """Storage of a single model with metadata."""

    def __init__(
        self, net, repetition: int, scheme: list, tracker: list, best_iter: int
    ):
        """Initializes a model object with general properties.

        Args:
            net (str): The string representation of the final trained network.
            repetition (int): Indicates the model repetition to which the network belongs.
            scheme (list): The layer scheme used for the network structure.
            tracker (list): The full error tracker of the training process.
            best_iter (int): The iteration considered the best, resulting in the trained 'net'.
        """
        self.net = net
        self.repetition = repetition
        self.scheme = scheme
        self.tracker = tracker
        self.best_iter = best_iter

    @property
    def error(self) -> float:
        """ Returns the error of the selected network. """
        return self.tracker[self.best_iter][0]

    def validation(self, valid: str = "train") -> float | None:
        """ Returns the validation achieved by the best network

        Args:
            valid (str): The type of validation, either "train", "test", or the "product" of both.
        """
        if valid == "train":
            return self.tracker[self.best_iter][1]
        elif valid == "test":
            return self.tracker[self.best_iter][2]
        elif valid == "product":
            return self.tracker[self.best_iter][3]
        return None

    def __str__(self) -> str:
        """ String representation of the stored data."""
        string = f"Repetition {self.repetition} for scheme {self.scheme}:\n"
        string += f"\tIteration {self.best_iter}\n"
        string += f"\tError: {self.error}\n"
        string += f"\tValidation train: {self.validation("train")}\n"
        string += f"\tValidation test: {self.validation("test")}\n"
        string += f"\tValidation product: {self.validation("product")}\n"
        return string


class ModelContainer:
    """Container to store fitted models."""

    def __init__(
        self,
        inputs: list,
        outputs: list,
        hl_schemes: list[list],
        LR: float,
        momentum: float | list,
        repetitions: int,
        optim: str,
        testpercent: float,
        validation: str,
    ):
        """Initializes the model container with specified architecture and training parameters.

        Args:
            inputs (list): The names of input features by the order given to model.
            outputs (list): The names of output classes as predicted by model.
            hl_schemes (list[list]): A list of lists defining the hidden layer structures (neurons per layer).
            LR (float): The learning rate for the training process.
            momentum (float | list): Momentum value(s) for the optimization algorithm; can be a single value or a list.
            repetitions (int): The number of times the model will be trained (repeated runs).
            optim (str): The optimization algorithm to use. Available methods are "SGD" (stochastic gradient descend), "SimpleMomentum", "Momentum", "Adagrad", "RMSProp" and "Adam".
            testpercent (float): The percentage of data to be used for testing.
            validation (str): The validation metric used to quantify network performance.
        """
        self.models = []
        self.inputs = inputs
        self.outputs = outputs
        self.hl_schemes = hl_schemes
        self.lr = LR
        self.momentum = momentum
        self.repetitions = repetitions
        self.testpercent = testpercent
        self.optimizer = optim
        self.validation_metric = validation
        self.version = __version__

    def __len__(self) -> int:
        """ Returns the number of networks in the container."""
        return len(self.models)

    def __str__(self) -> str:
        """String representation providing a summary of the classifier parameters and trained networks."""

        string = "ANN supervised learning model\n"
        string += "Model inputs provided:\n"
        for input in self.inputs:
            string += f"\t - {input}"
        string += "Model outputs (targets):\n"
        for output in self.outputs:
            string += f"\t - {output}"
        string += "Hidden layer schemes:\n"
        for scheme in self.hl_schemes:
            string += f"\t- i:[{self.n_inputs}] | hl:{scheme} | o:[{self.n_outputs}]\n"
        string += f"{self.repetitions} repetitions for each scheme.\n"
        string += f"Validation metric: {self.validation_metric}\n"
        string += f"Optimizer: {self.optimizer}\n"
        n = len(self)
        if n == 0:
            string += "No models trained yet.\n"
        else:
            string += f"{n} networks trained.\n"
            for model in self.models:
                string += str(model)
        return string

    @property
    def n_inputs(self):
        """ The number of inputs given for each model."""
        return len(self.inputs)

    @property
    def n_outputs(self):
        """ The number of outputs predicted by each model."""
        return len(self.outputs)

    def add_model(self, model):
        """Store a model in the container."""
        if isinstance(model, Model):
            self.models.append(model)
        else:
            raise ValueError(f"Model must be an instance of Model.")

    def get_best_models(self) -> list:
        """Retrieve the best models."""
        return self.models

    def get_best_nets(self) -> list:
        """Retrieve the best networks (without associated metadata)"""
        nets = []
        for i in range(len(self)):
            nets.append(self.models[i].net)
        return nets

    def save(self, filename):
        """write the model container to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load the model container from a file."""
        with open(filename, "rb") as f:
            return pickle.load(f)
