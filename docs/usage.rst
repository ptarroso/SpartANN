.. role:: py(code)
    :language: python

.. role:: bash(code)
    :language: bash

Usage
=====

The usage of SpartANN can be categorized into two approaches:

    - Simple Mode: Users can utilize the provided command-line tools for training and predicting without requiring any interaction with Python code.

    - Advanced Mode: Advanced users can access the full functionality of SpartANN by integrating it directly as a Python module, allowing for greater flexibility and customization.

Simple Use
----------
The 'tools' directory contains two Python scripts that can be used independently from the command line. These tools do not require any Python programming skills and allow users to build and predict full models, provided the data is in the correct format.

The currently available tools are:

    - buildmodel.py: Trains a model using a labeled dataset and a raster, producing a model file as output.

    -predict.py: Uses a trained model to generate predictions for a raster image with the same spectral information as the raster used during training.

These tools can be used directly from the command line by providing the respective paths to the files. To check their basic usage and list of arguments, you can use the -h option:

.. code-block:: bash

    python tools/buildmodel.py -h
    python tools/predict.py -h

More details are available in the Examples section.

Advanced use
------------

SpartANN can be further configured to make full use of its functionalities. It provides a rich API for tailoring specific user cases, from reading tabular data for class definitions and interacting with raster images for training data extraction, to fully customizing Artificial Neural Network definitions. This makes SpartANN adaptable to a wide range of needs.

Although advanced usage requires some coding skills, SpartANN exposes an intuitive API, making it easy to access its core functionalities. You can start by importing SpartANN into your Python code as follows:

.. code-block:: python

    import spartann as sa

More details are provided in the Examples section.
