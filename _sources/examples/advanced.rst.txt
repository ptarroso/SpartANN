.. role:: py(code)
    :language: python

Advanced
========

In this advanced example, we interact with SpartANN in Python as a module. This approach provides more flexibility to adapt to different user cases and allows better configuration of the model definitions.

If SpartANN was installed via conda, the environment needs to be activated beforehand. Refer to the :ref:`Installation` section for further details.

To begin, we need to import the module into a Python session:

.. code-block:: py

    import spartann as sa


Read data
---------

SpartANN provides functions to read tabular data in text format (e.g., CSV) and raster data, and it allows the intersection of both to extract and prepare training data.

The point location data is read as a `DataTable`:

.. code-block:: py

    pnt = sa.DataTable.from_file("examples/data/train_data.csv", sep=";")
    print(pnt)


.. code-block:: text

    X|Y|Clouds
    561109.6|4495044.6|0
    565530.8|4496869.0|1
    561844.0|4494916.3|0
    562007.2|4499194.7|0
    562642.6|4498192.1|1
    566504.2|4496580.5|1
    567055.0|4494047.8|0
    564507.8|4497198.3|1
    562593.0|4497836.6|1
    ...

The `DataTable` is very flexible. You can easily add or remove locations and classes for the training dataset. While this is outside the scope of this example, you can explore the methods available in the :ref:`spartann.datatable package` to understand its full functionality.

Reading raster data is just as straightforward:

.. code-block:: py

    rst = sa.Raster.from_file("examples/data/Sentinel2_clouds.tif")
    rst.bandnames

.. code-block:: text

    ['B4, central wavelength 665 nm', 'B3, central wavelength 560 nm', 'B2, central wavelength 490 nm', 'B8, central wavelength 842 nm', 'B5, central wavelength 705 nm', 'B6, central wavelength 740 nm', 'B7, central wavelength 783 nm', 'B8A, central wavelength 865 nm', 'B11, central wavelength 1610 nm', 'B12, central wavelength 2190 nm', 'B1, central wavelength 443 nm', 'B9, central wavelength 945 nm', 'B10, central wavelength 1375 nm']

The `Raster` object uses the `osgeo/GDAL` library to open the dataset. It provides an intuitive set of methods to work with rasters interacting with the library. One key advantage is its integration with the `DataTable` for data extraction.

We can extract data from the raster at coordinate locations. Note that there is no coordinate reference system check, so all data must share the same reference system. Otherwise, the `DataTable` will only return nodata values.


.. code-block:: py

    pnt.getDataFromRaster(rst)
    print(pnt)


.. code-block:: text

    X|Y|Clouds|B4, central wavelength 665 nm|B3, central wavelength 560 nm|B2, central wavelength 490 nm|B8, central wavelength 842 nm|B5, central wavelength 705 nm|B6, central wavelength 740 nm|B7, central wavelength 783 nm|B8A, central wavelength 865 nm|B11, central wavelength 1610 nm|B12, central wavelength 2190 nm|B1, central wavelength 443 nm|B9, central wavelength 945 nm|B10, central wavelength 1375 nm
    561109.6|4495044.6|0|1422.0|1629.0|1941.0|2147.0|1449.0|1875.0|2022.0|1962.0|1423.0|1190.0|2304.0|1354.0|1013.0
    565530.8|4496869.0|1|3609.0|3367.0|3545.0|4493.0|4020.0|4612.0|4984.0|5245.0|4727.0|4670.0|3674.0|2256.0|1027.0
    561844.0|4494916.3|0|1455.0|1669.0|1946.0|2632.0|1510.0|1991.0|2143.0|2305.0|1608.0|1294.0|2306.0|1417.0|1012.0
    562007.2|4499194.7|0|1330.0|1527.0|1854.0|1519.0|1332.0|1494.0|1571.0|1580.0|1184.0|1079.0|2264.0|1211.0|1012.0

Now the points include the associated data with respective band names found in the raster.

For training the network, it is better to scale the data:

.. code-block:: py

    pnt.scaleData()

This centers and scales every variable in the `DataTable` for optimal training performance. The scale values used (means and standard deviations) are retained for transforming values during prediction.

The dataset is now ready for training.

Training the models
-------------------

The first step in our process is to define a series of parameters for the modeling approach. These include the network architecture, learning parameters, and the strategy for producing multiple models.

In the example outlined in :ref:`A First Model`, we were limited to using only one architecture. Here, we have the flexibility to ensemble multiple network complexities. We will define three architectures with increasing levels of complexity. Simple networks tend to overgeneralize predictions, while complex networks often overfit the data. Ensembling these networks provides a balanced prediction by leveraging their different strengths.

We will use the following architectures:

- **Simple**: This architecture consists of a single hidden layer with 7 neurons, represented as `[7]`.
- **Medium**: This replicates the architecture from the earlier example, with three layers consisting of 10, 6, and 3 neurons in each layer, represented as `[10, 6, 3]`.
- **Complex**: For this example, we use a more intricate architecture with five hidden layers, represented as `[10, 8, 8, 6, 3]`. While this architecture is likely overkill for the problem, it helps illustrate the range of complexity.

For each architecture, we will generate 5 repetitions, resulting in a total of `5 * 3 = 15` predictions per pixel. In each repetition, 20% of the data will be reserved for testing the network.

We will change the default learning optimizer from *RMSProp* (the only option in the previous example) to *Adam*. The *Adam* optimizer requires the definition of two momentum parameters. We will use a learning rate of 0.01. Since *Adam* is adaptive, the learning rate will vary throughout the iterative process. The two momentum parameters will be set to 0.9 and 0.999, as is typical for this optimizer.

.. code-block:: py

    rep = 5
    test = 20
    schemes = [[7], [10,6,3], [10, 8, 8, 6, 3]]
    LRate=0.001
    mom=[0.9, 0.999]
    optim = "Adam"

We can proceed with the training stage by initializing a classifier and training the networks:

.. code-block:: py

    ann = sa.AnnClassifier.from_datatable(pnt,
        repetitions=rep,
        testpercent=test,
        hl_schemes = schemes,
        LR=LRate,
        momentum=mom,
        optim=optim)

    ann.trainModel()

.. code-block:: text

    Training networks with scheme i:[13] | hl:[7] | o:[1].
    Repetition: 1 from 5
    | Iteration |   Error   | Train |  Test | Product |  ErrDiff |
    |      1073 |   1.72282 | 0.790 | 0.667 |  0.527  |  0.00081 |
    Best net:
    	Iteration 945
    	Error: 1.832
    	Validation train: 0.790
    	Validation test: 0.667
    	Validation product: 0.527
        ...

The iterative process displays the current repetition and network scheme. It also indicates the number of iterations required to minimize the best result, based on the optimal combination of train and test performance.

We can inspect the built models using the following code:


.. code-block:: py

    print(ann)

.. code-block:: text

    ANN supervised learning model
    Model inputs provided:
    	- B4, central wavelength 665 nm
      	- B3, central wavelength 560 nm
        - B2, central wavelength 490 nm
        - B8, central wavelength 842 nm
        - B5, central wavelength 705 nm
        - B6, central wavelength 740 nm
        - B7, central wavelength 783 nm
        - B8A, central wavelength 865 nm
        - B11, central wavelength 1610 nm
        - B12, central wavelength 2190 nm
        - B1, central wavelength 443 nm
        - B9, central wavelength 945 nm
        - B10, central wavelength 1375 nm
    Model outputs (targets):
    	- Clouds
    Hidden layer schemes:
    	- i:[13] | hl:[7] | o:[1]
    	- i:[13] | hl:[10, 6, 3] | o:[1]
    	- i:[13] | hl:[10, 8, 8, 6, 3] | o:[1]
    5 repetitions for each scheme.
    Validation metric: Cohen's Kappa
    Optimizer: Adam
    15 networks trained.
    Repetition 0 for scheme [13, 7, 1]:
       	Iteration 945
    	Error: 1.8319342549497482
    	Validation train: 0.7902097902097902
    	Validation test: 0.6666666666666666
    	Validation product: 0.5268065268065267
    Repetition 1 for scheme [13, 7, 1]:
    	Iteration 133
    	Error: 2.849658052782215
    	Validation train: 0.701067615658363
    	Validation test: 0.5
    	Validation product: 0.3505338078291815
    Repetition 2 for scheme [13, 7, 1]:
    	Iteration 129
    	Error: 2.810948597977949
    	Validation train: 0.701067615658363
    	Validation test: 0.5263157894736842
    	Validation product: 0.3689829556096647
    Repetition 3 for scheme [13, 7, 1]:
    	Iteration 643
    	Error: 2.0116930046623356
    	Validation train: 0.781021897810219
    	Validation test: 0.4
    	Validation product: 0.3124087591240876
    Repetition 4 for scheme [13, 7, 1]:
    	Iteration 777
    	Error: 1.304985786439687
    	Validation train: 0.9166666666666666
    	Validation test: 0.4
    	Validation product: 0.3666666666666667
    Repetition 0 for scheme [13, 10, 6, 3, 1]:
    	Iteration 930
    	Error: 1.0263790532540662
    	Validation train: 0.9581881533101045
    	Validation test: 0.8333333333333334
    	Validation product: 0.7984901277584204
    Repetition 1 for scheme [13, 10, 6, 3, 1]:
    	Iteration 125
    	Error: 4.291481186293516
    	Validation train: 0.6502732240437158
    	Validation test: 0.6666666666666666
    	Validation product: 0.43351548269581053
    Repetition 2 for scheme [13, 10, 6, 3, 1]:
    	Iteration 1099
    	Error: 1.324294339644675
    	Validation train: 0.9162303664921466
    	Validation test: 1.0
    	Validation product: 0.9162303664921466
    Repetition 3 for scheme [13, 10, 6, 3, 1]:
    	Iteration 1040
    	Error: 1.7438099420979907
    	Validation train: 0.8309859154929577
    	Validation test: 0.8235294117647058
    	Validation product: 0.6843413421706711
    Repetition 4 for scheme [13, 10, 6, 3, 1]:
    	Iteration 1131
    	Error: 0.6258572183908
    	Validation train: 0.9578947368421052
    	Validation test: 0.8333333333333334
    	Validation product: 0.7982456140350878
    Repetition 0 for scheme [13, 10, 8, 8, 6, 3, 1]:
    	Iteration 722
    	Error: 1.3057673879941003
    	Validation train: 0.8745644599303136
    	Validation test: 0.47058823529411764
    	Validation product: 0.41155974584955934
    Repetition 1 for scheme [13, 10, 8, 8, 6, 3, 1]:
    	Iteration 41
    	Error: 5.981142437982789
    	Validation train: 0.5352112676056338
    	Validation test: 1.0
    	Validation product: 0.5352112676056338
    Repetition 2 for scheme [13, 10, 8, 8, 6, 3, 1]:
    	Iteration 1373
    	Error: 1.7830216289715917
    	Validation train: 0.9154929577464789
    	Validation test: 1.0
    	Validation product: 0.9154929577464789
    Repetition 3 for scheme [13, 10, 8, 8, 6, 3, 1]:
    	Iteration 769
    	Error: 1.5401543510961322
    	Validation train: 0.9162303664921466
    	Validation test: 0.6666666666666666
    	Validation product: 0.6108202443280977
    Repetition 4 for scheme [13, 10, 8, 8, 6, 3, 1]:
    	Iteration 1137
    	Error: 1.7273137773705625
    	Validation train: 0.8309859154929577
    	Validation test: 0.8235294117647058
    	Validation product: 0.6843413421706711

We can write the models to a file so we can retrieve it later for predictions in same or different images (with the same bands).

.. code-block:: py

    ann.writeModel("Clouds_model.obj")


Predicting with model
---------------------

If you kept the session open, you will need to generate an `AnnPredict` object from the classifier. Since the raster to predict is the same, it does not need to be reopened.

.. code-block:: py
    ap = sa.AnnPredict.from_annclassifier(ann)

If you **restarted the Python session**, you can retrieve the saved models and the raster. To do so, start a new Python session, import the required modules, and load the models and raster.

.. code-block:: py

    import spartann as sa
    ap = sa.AnnPredict.from_modelsfile("Clouds_model.obj")
    rst = sa.Raster.from_file("examples/data/Sentinel2_clouds.tif")


With SpartANN, you can predict using a list of values, a `DataTable`, or a raster. Since predicting with a raster is the most common use case, the following example demonstrates this. Note that you can specify multicores to this function, allowing to leverage the use of multiple CPUs for predictions:

.. code-block:: py

    pred = ap.predictFromRaster(rst, ncores=5)

The prediction process might take some time, as it computes values for all pixels. Once the prediction is complete, you can save the raster to a TIFF file for inspection in any GIS software.

.. code-block:: py

    pred.writeRaster("Results.tif")

The band names in the output raster are descriptive, indicating which repetition and scheme were used for each prediction.

.. code-block:: py

    for i, bname in enumerate(pred.bandnames):
        print("band", i+1, ":", bname)

.. code-block:: text

    band 1 : Prediction for Clouds, with scheme [13, 7, 1], repetition 0
    band 2 : Prediction for Clouds, with scheme [13, 7, 1], repetition 1
    band 3 : Prediction for Clouds, with scheme [13, 7, 1], repetition 2
    band 4 : Prediction for Clouds, with scheme [13, 7, 1], repetition 3
    band 5 : Prediction for Clouds, with scheme [13, 7, 1], repetition 4
    band 6 : Prediction for Clouds, with scheme [13, 10, 6, 3, 1], repetition 0
    band 7 : Prediction for Clouds, with scheme [13, 10, 6, 3, 1], repetition 1
    band 8 : Prediction for Clouds, with scheme [13, 10, 6, 3, 1], repetition 2
    band 9 : Prediction for Clouds, with scheme [13, 10, 6, 3, 1], repetition 3
    band 10 : Prediction for Clouds, with scheme [13, 10, 6, 3, 1], repetition 4
    band 11 : Prediction for Clouds, with scheme [13, 10, 8, 8, 6, 3, 1], repetition 0
    band 12 : Prediction for Clouds, with scheme [13, 10, 8, 8, 6, 3, 1], repetition 1
    band 13 : Prediction for Clouds, with scheme [13, 10, 8, 8, 6, 3, 1], repetition 2
    band 14 : Prediction for Clouds, with scheme [13, 10, 8, 8, 6, 3, 1], repetition 3
    band 15 : Prediction for Clouds, with scheme [13, 10, 8, 8, 6, 3, 1], repetition 4


Results
-------

SpartANN does not provide built-in plotting capabilities for the results, but it can easily interact with external libraries such as matplotlib (not a listed requirement) for visualization. Alternatively, you can use any GIS software or other programming environments, such as R, for further analysis and visualization.

Below, we showcase the results produced without including the code.

The predictions show slight differences in their ability to detect clouds, but the core output remains consistent regardless of the repetition or network scheme.

.. image:: assets/advanced_predictions.png

By calculating the mean and standard deviation, we can gain a clearer understanding of the core predictions and the associated uncertainty. We can use the :code:`Raster.aggregate_bands()` method for this purpose. We have to define a function for aggregation that accepts the argument :code:`axes`. The most easy functions are the numpy functions such as :code:`np.mean` and :code:`np.std`, but you can aggregate with any. We will create mean prediction an standard deviation of the predictions, which will create two new rasters that we can write to file. (Note: as following GDAL convention, bands start at index 1)

.. code-block:: py

    import numpy as np
    bands = [i+1 for i in range(res.nbands)]
    rst_mean = res.aggregate_bands(bands, fun=np.mean)
    rst_sdev = res.aggregate_bands(bands, fun=np.std)
    rst_mean.writeRaster("Mean_prediction.tif")
    rst_sdev.writeRaster("SDev_prediction.tif")

We can see both results:

.. image:: assets/advanced_ensemble.png

When superimposed onto the original raster, the success of the cloud detection process becomes evident:

.. image:: assets/advanced_predicted_clouds.png
