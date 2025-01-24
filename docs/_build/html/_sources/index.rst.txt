.. SpartANN documentation master file, created by
   sphinx-quickstart on Wed Jan 15 16:56:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SpartANN
========

**Spectral Pattern Analysis and Remote-sensing Tool with Artificial Neural Networks**

SpartANN is a user-friendly Python tool for remote sensing classification, designed to simplify the analysis of raster imagery based on tabular data. Built with Artificial Neural Networks (ANN), it excels at identifying non-linear associations between training data and classification targets, delivering high-quality predictions.

At its core, SpartANN uses a fully connected Multilayer Perceptron network with a customizable structure. The network can range from a simple design with a single hidden layer to complex deep-learning configurations with multiple hidden layers and user-defined numbers of neurons. This flexibility allows SpartANN to adapt to diverse classification tasks and user needs.

SpartANN is designed for ease of use with common data formats. Typical inputs include text files for tabular data and raster data in the standard TIFF format, though other formats are also supported. Its primary strength lies in simplifying model building and mitigating overfitting through ensemble predictions. These ensembles are generated from repetitions of the same network structure or by combining predictions from different network configurations.

For basic use, SpartANN provides a straightforward structure for importing tabular data and raster imagery, extracting training data directly from the raster. Users can save trained models and apply them to make predictions on new imagery. Additionally, SpartANN supports both categorical and continuous targets, offering single or multiple predictions for each target category.

Because SpartANN is a Python module, advanced customization and integration into broader workflows are also possible, making it a versatile tool for remote sensing tasks.

.. toctree::
   :maxdepth: 4
   :caption: Getting Started

   Installation <install>
   Usage <usage>

.. include:: examples.rst

.. include:: modules.rst
