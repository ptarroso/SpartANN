from typing import Callable

def cohenkappa(real: list, pred: list) -> float:
    """Cohen's Kappa

    Calculates Cohen's Kappa from a list of real and predicted values.

    Args:
        real: a list of zeros and ones referring to binary targets.
        pred: a list of same size of real with predicted binary values.
    """
    TP = TN = FP = FN = 0
    for i in range(len(real)):
        TP += (real[i] + pred[i]) == 2
        TN += (real[i] + pred[i]) == 0
        FP += ((1 - real[i]) + pred[i]) == 2
        FN += (real[i] + (1 - pred[i])) == 2
    k = (2 * (TP * TN - FN * FP)) / ((TP + FP) * (FP + TN) + (TP + FN) * (FN + TN))
    return k


def pearsonscorr(real: list, pred: list) -> float:
    """Pearson's correlation

    Calculates Pearson's correlation from two list of continuous values.

    Args:
        real: the target values.
        pred: the predicted values by the model.

    """
    n = len(real)
    r_bar = sum(real) / n
    p_bar = sum(pred) / n
    sums = [0, 0, 0]
    for i in range(n):
        sums[0] += (real[i] - r_bar) * (pred[i] - p_bar)
        sums[1] += (real[i] - r_bar) ** 2
        sums[2] += (pred[i] - p_bar) ** 2
    return sums[0] / (sums[1] * sums[2]) ** 0.5

class Validation:
    """General class for providing validation metric to classifier."""

    def __init__(
        self,
        func: Callable[[list, list], float],
        name: str,
        threshold: None | float = None,
    ):
        """Initialise validation.

        Args:
            func: a function that returns a validation metric from two list arguments for the calibration data and predicted data
            name: the name of the validation metric
            threshold: either 'None' for continuous metrics or a float for thresholding predictive values.
        """
        self.func = func
        self.name = name
        self.threshold = threshold

    def calc(self, real: list[list], pred: list[list]) -> float:
        """Calculate metric. Used by classifier."""
        if self.threshold:
            pred = [(x > self.threshold) * 1 for y in pred for x in y]
        else:
            pred = [x for y in pred for x in y]
        real = [x for y in real for x in y]
        val = self.func(real, pred)
        return val

    def __str__(self):
        """Returns the name of the valitation metric."""
        return self.name



CohensKappa_Validation = Validation(cohenkappa, "Cohen's Kappa", 0.5)
"""Assess the performance of a binary classification model using Cohen's Kappa.

This instantiation of the "Validation" class simplifies the evaluation process
for the user. It is specifically designed for binary datasets and uses Cohen's
Kappa metric to measure agreement between predictions and calibration data.

Note:
- The default threshold for classification is set to 0.5, meaning values
  greater than or equal to 0.5 are classified as positive, and values
  below 0.5 are classified as negative.
"""

PearsonsCorr_Validation = Validation(pearsonscorr, "Pearson's Correlation")
"""Assess the performance of a continuous data using Pearson's Correlation Score.

This instantiation of the "Validation" class simplifies the evaluation process
for the user. It is specifically designed for continuous datasets and uses Pearson's
Correlation Score to measure agreement between predictions and calibration data.
"""
