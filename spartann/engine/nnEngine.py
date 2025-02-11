#!/usr/bin/env python
"""
FeedForward Backpropagation Neural Networks
Version 3.00
Copyright (C) 2010-2024  Pedro Tarroso

Originally published in
"Tarroso, P., Carvalho, S. & Brito, J.C. (2012) Simapse - Simulation
Maps for Ecological Niche Modelling. Methods in Ecology and Evolution
doi: 10.1111/j.2041-210X.2012.00210.x"

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from random import uniform, normalvariate, sample
from math import exp

__all__ = ["NN"]

class NN:
    def __init__(self, scheme=[], iterations=1000, LR=0.01, momentum=0.9, optim="RMSProp", verbosity=1, eps=1e-18):
        """ Initialise ANN.

        Args:
            scheme: a list with layers and neurons where length indicate number of layers (input and output included) and integers number of neurons per layer. A structure with 5 inputs, 2 hidden layers with 5 neurons in the first layer and 3 neurons in the second layer, plus a single output layer is [5,5,3,1].
            iterations: integer indicating the maximum number of iterations to train
            LR: float indicating learning rate.
            momentum: a float or list indicating momentum value.
            optim: a string indicating the optimizer to use. Available options are "SGD", "SimpleMomentum", "Momentum", "AdaGrad", "RMSProp" or "Adam".
            verbosity: verbosity level of the training process
            eps: a small value for avoiding division by zero (no need to change)
        """
        self.scheme = scheme

        # Create network
        self.structure(scheme)

        self.iterations = iterations
        self.LearningRate = LR
        self.momentum = momentum

        self.varNames = None
        self.errPat = None
        self.netTrainError = None

        # Scaling parameters
        self.means = None
        self.sdevs = None

        self.func = sigm
        self.dfunc = dsigm

        self.verbosity = verbosity
        avail_optimizers = ["sgd", "simplemomentum", "momentum", "adagrad", "rmsprop", "adam"]
        if optim.lower() in avail_optimizers:
            if optim.lower() == "adam" and not isinstance(self.momentum, list):
                raise ValueError("Adam optimizer needs two parameters. Set momentum as a list with [B1, B2].")
            self.optimizer = optim.lower()
        else:
            raise ValueError(f"Optimizer {optim} not available.")
        self.eps = eps


    @classmethod
    def loadnet(cls, netstr):
        """Inits object with a string.

        The net string is the string representation of the NN trained network.

        Returns:
            NN: a trained network
        """
        netvars = {}
        for line in netstr.split("\n"):
            line = line.split(";")
            if line[0] == "optimizer":
                netvars[line[0]] = line[1]
            else:
                netvars[line[0]] = eval(line[1])
        net = cls(netvars["scheme"])
        net.__dict__ = netvars
        return net

    @classmethod
    def loadnet_fromfile(cls, netfile):
        """Inits object with a network in a file

        It opens a file containing the NN trained network details resulting from using
        NN.savenet().

        Returns:
            NN: a trained network

        """
        f = open(netfile, "r")
        netstr = f.read()
        f.close()
        return cls.loadnet(netstr)

    def __str__(self):
        """ String representation of the current network state.

            Returns:
                string: a network representation in text format
        """
        string = ""
        netvars = vars(self)
        nl = ""
        for var in netvars:
            value = netvars[var]
            if hasattr(value, "__call__"):
                value = value.__name__
            string += f"{nl}{var};{value}"
            nl = "\n"
        return string

    def savenet(self, netfile):
        """Saves the trained network to a file.

        It saves the current state of the network to a file.

        """
        f = open(netfile, "w")
        f.write(str(self))
        f.close()

    def setFunc(self, fun, dfun):
        """ Sets activation function and derivative function

        Allows to change the neuron activation function and derivative. The default initialization
        sets the sigmoid and respective derivative function for activation. This funtion allows to
        change the activation with the pair of functions.

        Args:
            fun: A function accepting a float and return other based on desired activation response
            dfun: The respective derivative function of fun that accepts and returns a float.

        """
        self.func = fun
        self.dfunc = dfun

    @property
    def means(self):
        """Returns the vector of means used for scaling inputs."""
        return self._means

    @means.setter
    def means(self, values):
        """Sets the vector of means used for scaling inputs."""
        if values is None:
            self._means = None
        else:
            if len(values) == self.scheme[0]:
                self._means = values
            else:
                raise ValueError(
                    "List length does not match the number of input neurons."
                )

    @property
    def sdevs(self):
        """Returns the vector of standard deviations used for scaling inputs."""
        return self._sdevs

    @sdevs.setter
    def sdevs(self, values):
        """Sets the vector of standard deviations used for scaling inputs."""
        if values is None:
            self._sdevs = None
        else:
            if len(values) == self.scheme[0]:
                self._sdevs = values
            else:
                raise ValueError(
                    "List length does not match the number of input neurons."
                )

    def trainnet(self, patterns, targets, batch_size=1, varnames=None, scale=True, verbose=None):
        """Trains the network with loaded inputs.

        Args:
            patterns (list[list]): a list of n training data with a list of p input (i) values to classify
            [[i_1_1, i_2_1, ..., i_p_1], [i_1_2, i_2_2, ...,i_p_2], ..., [i_1_n, i_2_n, ..., i_p_n]]
            targets (list[list]): a list of n classification targets for training with a list of o outputs
            [[O_1_1, O_2_1, ..., O_o_1], [O_1_2, O_2_2, ...,O_o_2], ..., [O_1_n, O_2_n, ..., O_o_n]]
            varnames (list): a list of variable names for each input
            scale (bool): if True scales the inputs to z-scores (x-mean(X))/sdev(X))
            verbose (bool): Control the verbosity of training.

        Verbose level
            0 - Nothing is printed
            1 - Prints Iteration number | Network error

        """
        nsamples = len(patterns)

        if batch_size and (batch_size > nsamples or batch_size <= 0):
            msg = "Batch size must be greater than 0 and less or equal to number of patterns."
            raise ValueError(msg)

        if verbose is None:
            verbose = self.verbosity

        # Scaling
        if scale:
            if self.means is not None and self.sdevs is not None:
                Warning(
                    "Means and Standard deviations are already set. Using for scaling input patterns"
                )
            else:
                self.__calcstats(patterns)
            patterns = self.scale(patterns)

        self.varnames = varnames
        # train with patterns and targets
        for i in range(self.iterations):
            self.cur_iter = i
            for p in range(nsamples):
                pattern = patterns[p][:]
                target = targets[p]

                # Calculates the current network output
                # and error for this pattern
                # FEED FORWARD
                self.feedforward(pattern)
                # Calculate net error
                self.calcTargetError(target)
                # Change weights on network

                if p % batch_size + 1 == batch_size:
                    self.update_weights(batch_size)

                # BACK PROPAGATION
                self.backpropag(pattern)

            if nsamples % batch_size > 0:
                self.update_weights(nsamples % batch_size)

            err = self.neterror(patterns, targets, "SSerror")
            self.netTrainError = err
            # Print error for this iteration
            if verbose == 1:
                print("iteration = {} | error = {}".format(i, err))
        self.cur_iter = None

    def checkVarNames(self, varnames):
        """Tests if varnames are the same an in the same order as trained"""
        if self.varNames is None:
            # No variable names provided: the user must control the order.
            return True
        if len(self.varNames) == len(varnames):
            test = [x[0] == x[1] for x in zip(self.varNames, varnames)]
            if sum(test) == len(test):
                # Variable names match
                return True
            else:
                # some variables do not match
                return False
        else:
            # diferent number of variables
            return False

    def __calcstats(self, patterns):
        """Calculate statistics about inputs.

        Calculates and retains the vector of means and standard deviation of the inputs used for traning.

        """
        nvar = self.scheme[0]
        n = len(patterns)

        # Variable average
        means = [0.0] * nvar
        for i in range(n):
            means = [means[x] + patterns[i][x] for x in range(nvar)]
        means = [means[x] / n for x in range(nvar)]

        # Variable standard deviation
        stdevs = [0.0] * nvar
        for i in range(n):
            stdevs = [stdevs[x] + (patterns[i][x] - means[x]) ** 2 for x in range(nvar)]
        stdevs = [stdevs[x] / n for x in range(nvar)]

        # Assign self variables
        self.means = means
        self.sdevs = stdevs

    def scale(self, patterns):
        """Scale given patterns with stored means and standard deviations

        This function scales a set of patterns with stored means and standard deviations.
        It is useful for predicting using the same scaling values.

        Args:
            patterns (list([list]): a list of input patterns similar to training or testing.

        Returns:
            list: a list of scaled values with same dimensions as patterns.

        """
        nvar = self.scheme[0]
        n = len(patterns)
        scaled = [
            [(patterns[i][j] - self.means[j]) / self.sdevs[j] for j in range(nvar)]
            for i in range(n)
        ]
        return scaled

    def testnet(self, patterns, varnames=None, scale=True, verbose=None):
        """Tests the network with a sequence of patterns and returns
        the predicted output.

        Args:
            patterns (list[list]): a list of n testing data with a list of p input (i) values to classify
            [[i_1_1, i_2_1, ..., i_p_1], [i_1_2, i_2_2, ...,i_p_2], ..., [i_1_n, i_2_n, ..., i_p_n]]
            varnames (list): a list of variable names for each input
            scale (bool): if True scales the inputs to z-scores (x-mean(X))/sdev(X))
            verbose (bool): Control the verbosity for testing.

        Verbose level
            0 - nothing is printed
            1 - prints \'pattern number | network output\'

        Returns:
            list[list]: a list of same length as patterns with the required number of output predictions.
        """

        if not self.checkVarNames(varnames):
            raise ValueError("Variable names do not match variables used for training.")

        if verbose is None:
            verbose = self.verbosity
        # Accepts a single list of inputs or a list of sequence of inputs
        if not isinstance(patterns[0], list):
            patterns = [patterns]

        if scale:
            patterns = self.scale(patterns)

        nPat = len(patterns)
        finalresults = []
        for p in range(nPat):
            curPat = patterns[p]
            self.feedforward(curPat)
            result = self.values[-1][:]
            finalresults.append(result)
            if verbose == 1:
                print("Pattern = {} | predicted = {}".format(p + 1, result))

        return finalresults

    def neterror(self, patterns, targets, errorType="SSerror"):
        """Calculates the overall error of the network.

        The error of the network based on a set of patterns and targets,
        usually the training data or for some independent evaluation of the
        fit.

        Args:
            patterns (list[list]): a list of n training data with a list of p input (i) values to classify
            [[i_1_1, i_2_1, ..., i_p_1], [i_1_2, i_2_2, ...,i_p_2], ..., [i_1_n, i_2_n, ..., i_p_n]]
            targets (list[list]): a list of n classification targets for training with a list of o outputs
            [[O_1_1, O_2_1, ..., O_o_1], [O_1_2, O_2_2, ...,O_o_2], ..., [O_1_n, O_2_n, ..., O_o_n]]
            errorType (string) :  Options for error type are:
                - RMSerror - Root Mean Square error (default)
                - SSerror  - Sum of Squared error

        Returns:
            (float): the net error
        """
        try:
            if errorType not in ["RMSerror", "SSerror"]:
                raise SyntaxError("Error type must be 'RMSerror' or 'SSerror'!")

            if errorType == "RMSerror":
                error = self.__RMSerror(patterns, targets)
            elif errorType == "SSerror":
                error = self.__SSerror(patterns, targets)

            return error

        except SyntaxError as e:
            print(e)

    def __RMSerror(self, patterns, targets):
        """Calculates the Root Mean Square Error os the network."""
        nOutputs = self.scheme[-1]
        temp = [0.0] * nOutputs
        for p in range(len(patterns)):
            self.feedforward(patterns[p])
            self.calcTargetError(targets[p])
            temp = [temp[x] + (self.errPat[x]) ** 2 for x in range(nOutputs)]
        RMSerror = [(temp[x] / len(patterns)) ** 0.5 for x in range(nOutputs)]
        return RMSerror

    def __SSerror(self, patterns, targets):
        """Calculates the Sum of Squared Errors of the network."""
        nOutputs = self.scheme[-1]
        temp = [0.0] * nOutputs
        for p in range(len(patterns)):
            self.feedforward(patterns[p])
            self.calcTargetError(targets[p])
            temp = [temp[x] + (self.errPat[x]) ** 2 for x in range(nOutputs)]
        SSerror = [0.5 * (temp[x]) for x in range(nOutputs)]
        return SSerror

    def _optimSGD(self, l, n, w):
        """ Implementation of Simple Gradient Descent optimizer."""
        grad = self.G[l][n][w][0]
        self.changes[l][n][w] = self.LearningRate * grad

    def _optimSimpleMomentum(self, l, n, w):
        """ Implementation of SGD with simple momentum optimizer"""
        grad = self.G[l][n][w][0]
        self.changes[l][n][w] = self.LearningRate * grad + self.momentum * self.changes[l][n][w]

    def _optimMomentum(self, l, n, w):
        """ Implementation of SGD with momentum optimizer"""
        prevG = self.G[l][n][w][1]
        grad = self.G[l][n][w][0]
        self.changes[l][n][w] = self.LearningRate * ((1-self.momentum) * grad + (self.momentum * prevG))
        self.G[l][n][w][1] = grad

    def _optimAdagrad(self, l, n, w):
        """ Implementation of Adaptive Gradient Optimization (AdaGRAD) optimizer"""
        grad = self.G[l][n][w][0]
        self.G[l][n][w][1] += grad**2
        self.changes[l][n][w] =  self.LearningRate/((self.G[l][n][w][1]+self.eps)**0.5) * grad

    def _optimRMSProp(self, l, n, w):
        """ Implementation of Root Mean Squared Propagation (RMSProp) optimizer"""
        grad = self.G[l][n][w][0]
        self.G[l][n][w][1] = self.momentum * self.G[l][n][w][1] + (1-self.momentum)*(grad**2)
        self.changes[l][n][w] =  self.LearningRate/((self.G[l][n][w][1]+self.eps)**0.5) * grad

    def _optimAdam(self, l, n, w):
        """ Implementation of Adaptive Moment Estimation (Adam) optimizer"""
        B1,B2, = self.momentum
        grad = self.G[l][n][w][0]
        # Needs to track two values, so, it modifies G in the first iteration
        if self.G[l][n][w][1] == 0:
            self.G[l][n][w][1] = [0,0]
        self.G[l][n][w][1][0] = B1*self.G[l][n][w][1][0] + (1-B1)*grad
        self.G[l][n][w][1][1] = B2*self.G[l][n][w][1][1] + (1-B2)*(grad**2)
        grad_bias = self.G[l][n][w][1][0] / (1-B1**(self.cur_iter+1))
        gradsq_bias = self.G[l][n][w][1][1] / (1-B2**(self.cur_iter+1))
        self.changes[l][n][w] =  self.LearningRate/((gradsq_bias+self.eps)**0.5) * grad_bias

    def _getOptimizer(self):
        """Available optimizers

        Returns:
            a dictionary with implemented optimizers.

        """
        optimizers = {"sgd": self._optimSGD,
                      "simplemomentum": self._optimSimpleMomentum,
                      "momentum": self._optimMomentum,
                      "adagrad": self._optimAdagrad,
                      "rmsprop": self._optimRMSProp,
                      "adam": self._optimAdam}
        return optimizers[self.optimizer]

    def backpropag(self, patterns):
        """Backpropagation

        Backpropagates error calculation on neural network structure and updates weights.

        Args:
            patterns (list[list]): a list of n training data with a list of p input (i) values to classify
            [[i_1_1, i_2_1, ..., i_p_1], [i_1_2, i_2_2, ...,i_p_2], ..., [i_1_n, i_2_n, ..., i_p_n]]

        """
        scheme = self.scheme
        dfunc = self.dfunc
        nlayers = len(scheme)
        weights = self.weights
        values = self.values
        out_errors = self.errPat[:]
        grad = self.G

        # Calculate error at output level
        errors = [0.0] * (scheme[-2] + 1)
        for n in range(scheme[-1]):
            derivative = dfunc(values[-1][n])
            delta = derivative * out_errors[n]
            for w in range(scheme[-2]):
                grad[-1][n][w][0] += delta * values[-2][w]
                errors[w] += weights[-1][n][w] * out_errors[n]
            # Bias
            errors[-1] += weights[-1][n][-1] * out_errors[n]
            grad[-1][n][-1][0] += delta

        # Calculate error for hidden layers (except first)
        for l in range(nlayers - 2, 1, -1):
            prevL_errors = [0.0] * (scheme[l - 1] + 1)
            for n in range(scheme[l]):
                derivative = dfunc(values[l - 1][n])
                delta = derivative * errors[n]
                for w in range(scheme[l - 1]):
                    grad[l-1][n][w][0] += delta * values[l - 2][w]
                    prevL_errors[w] += weights[l - 1][n][w] * errors[n]
                # Bias
                prevL_errors[-1] += weights[l - 1][n][-1] * errors[n]
                grad[l-1][n][-1][0] += delta
            errors = prevL_errors[:]

        # Calculate error for the first hidden layer
        for n in range(scheme[1]):
            derivative = dfunc(values[0][n])
            delta = derivative * errors[n]
            for w in range(scheme[0]):
                grad[0][n][w][0] += delta * patterns[w]
            # Bias
            grad[0][n][-1][0] += delta

    def update_weights(self, batch_size):
        """Updates weights using averaged gradients"""
        optimizer = self._getOptimizer()
        weights = self.weights
        changes = self.changes
        grad = self.G

        for l in range(len(grad)):
            for n in range(len(grad[l])):
                for w in range(len(grad[l][n])):
                    grad[l][n][w][0] /= batch_size
                    optimizer(l, n, w)
                    weights[l][n][w] -= changes[l][n][w]
                    grad[l][n][w][0] = 0.0

    def feedforward(self, pattern):
        """Feedforward stage

        Calculates output values from a set of input patterns.

        Args:
            patterns (list[list]): a list of n training data with a list of p input (i) values to classify
            [[i_1_1, i_2_1, ..., i_p_1], [i_1_2, i_2_2, ...,i_p_2], ..., [i_1_n, i_2_n, ..., i_p_n]]

        """
        scheme = self.scheme
        hValues = self.values
        weights = self.weights
        nlayers = len(hValues)
        # Add bias node
        pattern = pattern + [1.0]
        derivatives = self.derivatives
        func = self.func
        dfunc = self.dfunc

        # FeedForward - Input patterns
        for n in range(scheme[1]):
            hValues[0][n] = 0.0
            for w in range(scheme[0]):
                hValues[0][n] += pattern[w] * weights[0][n][w]
            hValues[0][n] += weights[0][n][-1]
            hValues[0][n] = func(hValues[0][n])
            derivatives[0][n] = dfunc(hValues[0][n])

        # FeedForward - Hidden Layers and Output
        for l in range(1, nlayers):
            for n in range(scheme[l + 1]):
                hValues[l][n] = 0.0
                for w in range(scheme[l]):
                    hValues[l][n] += hValues[l - 1][w] * weights[l][n][w]
                hValues[l][n] += weights[l][n][-1]
                hValues[l][n] = func(hValues[l][n])
                derivatives[l][n] = dfunc(hValues[l][n])
        self.values = hValues

    def calcTargetError(self, target):
        """Calculates the error in relation to each output target."""
        hValues = self.values
        nOutputs = self.scheme[-1]
        errPat = [hValues[-1][o] - target[o] for o in range(nOutputs)]
        self.errPat = errPat

    def pderiv(self, patterns):
        """Processes the partial derivatives of the output vs. input
        Input patterns must be [[i1],[i2],[i3],...], or [patterns]"""
        if len(patterns) >= 1 and isinstance(patterns[0], list):
            pderiv = []
            for i in patterns:
                pderiv.append(self.__pderiv(i))
        elif len(patterns) == self.scheme[0] and isinstance(patterns[0], float):
            pderiv = self.__pderiv(patterns)
        elif len(patterns) < 1 or not isinstance(patterns[0], list):
            print("Input patterns must be [[i1],[i2],[i3],...] or [patterns]")
        return pderiv

    def _weight_gen(self, layer, rep=1, remove_bias=True):
        """Returns the weights of the layer sorted by the sequence of the
        neurons in the previous layer neurons. It repeats the sequence 'rep'
        number of times and it optionally removes the BIAS weight"""
        bias = int(bool(remove_bias))
        weights = self.weights
        new_weights = []
        w = weights[layer]
        for r in range(rep):
            for n_previous in range(len(w[0]) - bias):
                for n_next in range(len(w)):
                    new_weights.append(w[n_next][n_previous])
        return new_weights

    def __pderiv(self, pattern):
        """Returns the partial derivatives of the network output with respect
        to the input. This function only processes one sequence of input
        patterns by sprawling the network."""
        self.feedforward(pattern)

        deriv = self.derivatives
        weights = self.weights
        scheme = self.scheme[1:]
        nlayers = len(scheme)
        noutputs = scheme[-1]

        pderiv = [[0.0 for y in range(len(pattern))] for x in range(len(deriv[-1]))]
        for i in range(len(pattern)):
            # variable product has all the weights connected to input node i
            product = [weights[0][x][i] for x in range(len(weights[0]))]

            for l in range(nlayers - 1):  # output layer removed
                # n is the number of connections available to between current layers
                # m is number of nodes with derivatives in layer l
                n, m = len(product), len(deriv[l])
                # each deriv node value for layer l is repeated
                # by the number of connections it has (the same as the number of
                # neurons of previous layer)
                new_deriv = deriv[l] * int(n / m)
                # the products are multiplied by the derivatives sequentially
                product = [product[x] * new_deriv[x] for x in range(n)]
                # w has all the weights from next layer sorted by the neurons of
                # the previous layer and repeated for the number of connections
                # of the current calculation
                w = self._weight_gen(l + 1, n)
                # new product is created by repeating each value by the number of
                # neuron in the next layer
                sprawl_product = repeat(product, len(deriv[l + 1]))
                # sprawled product is multiplied by the sequence of weights between
                # current and next layer
                product = [sprawl_product[x] * w[x] for x in range(len(sprawl_product))]

            n, m = len(product), len(deriv[-1])
            new_deriv = deriv[-1] * int(n / m)
            product = [product[x] * new_deriv[x] for x in range(n)]

            for o in range(noutputs):
                for x in range(0, n, noutputs):
                    pderiv[o][i] += product[x + o]
        return pderiv

    def structure(self, network):
        """Creates the structure of the network

        Builds the network structure based on user definitions and sets all
        weights between neurons plus a BIAS neuron in each layer.

        Args:
            network (list): a network structure defining the number of layers
            and neurons in each layer. Number of layers is given by the length
            of the list and number of neurons is given by eacg integer item in
            the list. For instance [2, 4, 3, 1] creates a network with a input
            layer with two neurons, two hidden layers with 4 and 3 neurons each,
            and an output layer with a single output.

        """
        try:
            if self.scheme == []:
                raise SyntaxError("Network scheme cannot be empty!")

            nlayers = len(network)
            values, derivatives, weights, changes, G = [], [], [], [], []

            for layer in range(1, nlayers):
                values.append([])
                derivatives.append([])
                weights.append([])
                changes.append([])
                G.append([])
                nneurons, p_nneurons = network[layer], network[layer - 1] + 1
                i = layer - 1
                for neuron in range(nneurons):
                    values[i].append(0.0)
                    derivatives[i].append(0.0)
                    weights[i].append([0.0 for _ in range(p_nneurons)])
                    changes[i].append([0.0 for _ in range(p_nneurons)])
                    G[i].append([[0.0, 0.0] for _ in range(p_nneurons)])

            self.values = values
            self.derivatives = derivatives
            self.weights = weights
            self.changes = changes
            self.G = G

        except SyntaxError as e:
            print(e)

    def initWeights(self, method = "random", distrib="uniform"):
        """Network weigh initialization

        General function for weight initialization with a available methods:
            - Random: (method = "random") Recommende for small networks
            - Glorot/Xavier: (method = "glorot") Recommended for sigmoid or tanh activation
            - He: (method = "he") Recommended for ReLU activation (no dead neurons)
            - LeCun: (method = "lecun" ) Recommended for sigmoid functions and ReLU

        based on "normal" or "uniform" random numbers.
        (Note: for LeCun, distrib is always normal.)

        Args:
            method (string): One of the available methods.
            distrib (string): A distribution to be used.

        """
        methods = ["random", "glorot", "he", "lecun"]
        if method not in methods:
            raise ValueError(f"Method has to be one of {methods}.")
        method = methods.index(method)

        if method == 0:
            self.rndWeights(distrib)
        elif method == 1:
            self.glorotWeights(distrib)
        elif method == 2:
            self.heWeights(distrib)
        elif method == 3:
            self.lecunWeights()

    def rndWeights(self, distrib="uniform"):
        """
        Initiates all network weights with random numbers between [-1/sqrt(n_input), 1/sqrt(n_input)]
        based on uniform or normal distribution.
        """
        distrib = distrib.lower()
        if distrib not in ["uniform", "normal"]:
            raise ValueError("Distribution must be either 'uniform' or 'normal'.")
        w = self.weights
        for l in range(len(w)):
            rng = 1/(self.scheme[l])**0.5
            for n in range(len(w[l])):
                if distrib == "uniform":
                    w[l][n] = [uniform(-rng, rng) for x in w[l][n]]
                elif distrib == "normal":
                    w[l][n] = [normalvariate(0, rng) for x in w[l][n]]
        self.weights = w

    def glorotWeights(self, distrib="uniform"):
        """
        Initiates all network weights with random numbers between
        [-sqrt(6/(n_input+n_output)), sqrt(6/(n_input+n_output))] if uniform
        mu=0, sigma=2/(n_input+n_output), if normal
        """
        distrib = distrib.lower()
        if distrib not in ["uniform", "normal"]:
            raise ValueError("Distribution must be either 'uniform' or 'normal'.")
        w = self.weights
        for l in range(len(w)):
            nneur =self.scheme[l]+self.scheme[l+1]
            for n in range(len(w[l])):
                if distrib == "uniform":
                    rng = (6/nneur)**0.5
                    w[l][n] = [uniform(-rng, rng) for x in w[l][n]]
                elif distrib == "normal":
                    rng = 2/nneur
                    w[l][n] = [normalvariate(0, rng) for x in w[l][n]]
        self.weights = w

    def heWeights(self, distrib="uniform"):
        """
        Initiates all network weights with random numbers between
        [-sqrt(6/(n_input), sqrt(6/(n_input))] if uniform
        mu=0, sigma=2/(n_input), if normal
        """
        distrib = distrib.lower()
        if distrib not in ["uniform", "normal"]:
            raise ValueError("Distribution must be either 'uniform' or 'normal'.")
        w = self.weights
        for l in range(len(w)):
            nneur = self.scheme[l]
            for n in range(len(w[l])):
                if distrib == "uniform":
                    rng = (6/nneur)**0.5
                    w[l][n] = [uniform(-rng, rng) for x in w[l][n]]
                elif distrib == "normal":
                    rng = 2/nneur
                    w[l][n] = [normalvariate(0, rng) for x in w[l][n]]
        self.weights = w

    def lecunWeights(self):
        """
        Initiates all network weights with random normal numbers between
        mu=0, sigma=1/(n_input), if normal
        """
        w = self.weights
        for l in range(len(w)):
            nneur = self.scheme[l]
            for n in range(len(w[l])):
                rng = 1/nneur
                w[l][n] = [normalvariate(0, rng) for x in w[l][n]]
        self.weights = w

    def XORexample(self):
        """Example network data with XOR"""
        print("Loading XOR example...\n")
        Patterns = [[1, 0], [0, 1], [1, 1], [0, 0]]
        Targets = [[1.0], [1.0], [0.0], [0.0]]
        self.initWeights()
        self.trainnet(Patterns, Targets, scale=False, verbose=1)

        for p in range(len(Patterns)):
            self.feedforward(Patterns[p])
            self.calcTargetError(Targets[p])
            print(f"Pattern = {p+1} | real = {Targets[p]} | predicted = {self.values[-1]}")
        derivs = self.pderiv(Patterns)
        for i in range(len(Patterns)):
            print("Input: {} | Partial derivative: {}".format(Patterns[i], derivs[i]))


def repeat(lst, rep):
    """Creates a new list where each item is repeated 'rep'
    number of times, mantaining the original order:
    repeat([1,2,3], 2) = [1,1,2,2,3,3]"""
    new_lst = []
    for item in lst:
        for r in range(rep):
            new_lst.append(item)
    return new_lst

def tanh(x):
    result = None
    if x > 20:
        result = 1
    elif x < -20:
        result = -1
    else:
        a = exp(x)
        b = exp(x * -1)
        result = (a - b) / (a + b)
    return result

def dtanh(y):
    result = 1 - (y**2)
    return result

def sigm(x):
    e = 2.7182818284590451
    if x < -700:  # avoid overflow on large networks
        return 1 / (1 + e**700)
    else:
        return 1 / (1 + e ** (-x))

def dsigm(y):
    return y * (1 - y)

def relu(x):
    if x <0:
        return 0
    return x

def drelu(x):
    if x < 0:
        return 0
    return 1

if __name__ == "__main__":
    nn = NN([2, 3, 1], iterations=10000, LR=0.1, momentum=0.8)
    nn.XORexample()
