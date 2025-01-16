.. role:: py(code)
    :language: python

.. role:: bash(code)
    :language: bash

Installation
============

The installation of SpartANN is straightforward, as there are only a few requirements. The most recommended way to install SpartANN is through a conda environment. Requirements and environment files are provided along with the source code to simplify the process.

Whether you choose to install directly on your system or within a conda environment, the first step is to obtain the source code, which you can clone from the `GitHub repository <https://github.com/ptarroso/SpartANN>`_.

Requirements
------------

SpartANN has the following dependencies:

- `numpy <http://numpy.org/>`_
- `gdal <https://gdal.org/en/stable/api/python_bindings.html>`_

System Installation
-------------------

If you already have these dependencies installed on your system, installing SpartANN requires no additional steps. Simply navigate to the SpartANN directory in the cloned repository path and run:

.. code-block:: bash

    pip install .

Conda Installation
---------------------

This is the recommended installation method, as it minimizes the risk of interfering with your system environment and allows you to install specific versions of Python and the required dependencies. You will need a conda distribution, such as `miniconda <https://docs.anaconda.com/miniconda/install/>`_.

With conda installed, create the environment. Conda will automatically fetch the necessary packages and their dependencies. To create the environment, use the `environment.yml` file distributed with the source code. Open your terminal/command line, navigate to the SpartANN clone directory, and execute:

.. code-block:: bash

    conda env create --file environment.yml

Whenever you need to use SpartANN, activate the environment with:

.. code-block:: bash

    conda activate spartann

This will provide you with the correct Python version and the required modules to use SpartANN.

Check Installation
------------------

You can verify if SpartANN is correctly installed by attempting to import it in Python. Start Python and run the following:

.. code-block:: bash

    python

.. code-block:: python

    import spartann
    spartann.__version__

If the installation is successful, there should be no errors, and the installed SpartANN version will be displayed.
