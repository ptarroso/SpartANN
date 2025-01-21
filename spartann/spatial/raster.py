# -*- coding: utf-8 -*-
from __future__ import annotations
from osgeo import gdal, gdal_array
import numpy as np
import numbers
from math import ceil
from typing import Iterator, Callable, Tuple, Optional


gdal.UseExceptions()


def progressbar(current: float, total: float) -> None:
    """Wraper for GDAL style progress bar.

    Args:
        current: The current state
        total: The final state to be reached
    """
    _ = gdal.TermProgress_nocb(current / total)


class Raster(object):
    """Raster is a wrapper around GDAL raster IO.

    This class facilitates the usage of GDAL/osgeo raster in the contex of the
    classifier. It provides methods to simply read, create, modify and extract
    data from a raster.
    """

    def __init__(self, dts: gdal.Dataset) -> None:
        """Initiates a raster object from a gdal dataset

        Args:
            dts: a gdal Dataset object
        """
        self.dts = dts
        self.source = ""
    """ Size is [row, cols] which translates to [RasterY, RasterX]"""

    @classmethod
    def from_scratch(
        cls,
        size: Tuple[float, float],
        res: Tuple[float, float],
        crd: Tuple[float, float],
        bands: int = 1,
        nodata: float = np.nan,
        projwkt: Optional[str] = None,
        dtype: int = gdal.GDT_Float32,
    ):
        """Creates a new instance of Raster from scratch.

        Class method to create an empty raster by provinding a few descriptors.

        Args:
            size: List with row and col size. Note that [row, col] translates to [RasterYSize, RasterXSize].
            res: Spatial resolution of the raster given in [X,-Y] resolution.
            crd: Coordinate of the raster origin, usually the lower left corner
            bands: Number of bands to be created.
            nodata: No data value associated with the raster.
            projwkt: Well Known Text string describing the projection.
            dtype: Gdal Data type of the raster to be created.
        """
        driver = gdal.GetDriverByName("MEM")
        ds = driver.Create("", size[1], size[0], bands, dtype)
        if projwkt is not None:
            ds.SetProjection(projwkt)
        ds.SetGeoTransform([crd[0], res[1], 0, crd[1], 0, res[0]])
        for i in range(bands):
            ds.GetRasterBand(i + 1).Fill(nodata)
            ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
            ds.FlushCache()
        return cls(ds)

    @classmethod
    def from_file(cls, filename: str):
        """Creates a new instance of Raster from a raster file.

        Initializes a Raster object by reading a raster file

        Args:
            filename: a filename with full path to a raster file.

        """
        ds = gdal.Open(filename)
        rst = cls(ds)
        rst.source = filename
        return rst

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        res: Tuple[float, float],
        crd: Tuple[float, float],
        nodata: float|int = np.nan,
        projwkt: Optional[str] = None,
    ):
        """Creates a new instance of Raster from a numpy array.

        Args:
            arr: A 2D numpy array for a single band raster, or a 3D array for multiple bands. Shape of the array is (bands, rows, columns)
            res: Spatial resolution of the raster given in [X,-Y] resolution.
            crd: Coordinate of the raster origin, usually the lower left corner
            bands: Number of bands to be created.
            nodata: No data value associated with the raster.
            projwkt: Well Known Text string describing the projection.

        """
        driver = gdal.GetDriverByName("MEM")
        dtype = gdal_array.NumericTypeCodeToGDALTypeCode(arr.dtype)
        if len(arr.shape) == 2:
            arr = arr.reshape((1, arr.shape[0], arr.shape[1]))
        b,r,c = arr.shape
        ds = driver.Create("", c, r, b, dtype)
        if projwkt is not None:
            ds.SetProjection(projwkt)
        ds.SetGeoTransform([crd[0], res[1], 0, crd[1], 0, res[0]])
        ds.WriteArray(arr)
        _ = ds.GetRasterBand(1).SetNoDataValue(nodata)
        return cls(ds)

    @property
    def size(self) -> Tuple[float, float]:
        """Return the raster size [row, col]"""
        return (self.dts.RasterYSize, self.dts.RasterXSize)

    @property
    def res(self) -> Tuple[float, float]:
        """Returns the raster resolution [X,Y]"""
        gt = self.dts.GetGeoTransform()
        return (gt[5], gt[1])

    @property
    def origin(self) -> Tuple[float, float]:
        """Return the coordinate of the origin [X,Y]"""
        gt = self.dts.GetGeoTransform()
        return (gt[0], gt[3])

    @property
    def proj(self) -> str:
        """Returns the raster projection."""
        return self.dts.GetProjection()

    @property
    def nbands(self) -> int:
        """Returns the number of bands available in raster."""
        return self.dts.RasterCount

    @property
    def bandnames(self) -> list:
        """Returns a list with band names."""
        names = []
        for i in range(self.nbands):
            b = self.dts.GetRasterBand(i+1)
            names.append(b.GetDescription())
        return(names)

    @property
    def hasSource(self) -> bool:
        """ Indicates if the raster has a source file.

        If raster has source, it was read from or written to file.
        """
        return self.source != ""

    def addNewBand(self, data: np.ndarray, name: Optional[str] = None) -> None:
        """ Adds a new band to the raster datset and fill with data.

        Args:
            data: a numpy array for a single band and with same shape as base
            raster band (row size, col size)
            name: Optional name for band.
        """
        if data.shape != self.size:
            msg = f"Array must be of same size as raster {self.size}."
            raise ValueError(msg)

        _ = self.dts.AddBand(self.dts.GetRasterBand(1).DataType)
        self.set_array(data, band=self.nbands)

        if name:
            self.addDescription(name, self.nbands)

    def addMetadata(self, metadata_dict: dict, band: int | None = None) -> None:
        """Adds metadata to raster or raster band.

        Args:
            metadata_dict: a dictionary with items to add to metadata.
            band: band to add data (starts in 1) or 'None' for general metadata.
        """
        dts = self.dts
        if band:
            dts = self.dts.GetRasterBand(band)

        for item in metadata_dict:
            _ = dts.SetMetadataItem(f"{item}", f"{metadata_dict[item]}")

    def addDescription(self, description: str, band: int | None = None) -> None:
        """Adds a description to raster dataset or raster band.

        Args:
            description: a string with the text to add as description.
            band: band to add data (starts in 1) or 'None' for general dataset.
        """
        dts = self.dts
        if band:
            dts = self.dts.GetRasterBand(band)

        _ = dts.SetDescription(description)

    def getMetadata(self, band: int | None = None) -> dict:
        """Gets the metadata dict from raster or raster band."""
        dts = self.dts
        if band:
            dts = self.dts.GetRasterBand(band)

        return dts.GetMetadata()

    def getRowColFromXY(self,
        xy: Tuple[float, float]) -> Tuple[int|float, int|float]:
        """Converts coordinate pait into row,column."""
        gt = self.dts.GetGeoTransform()
        col = int((xy[0] - gt[0]) / gt[1])
        if col < 0 or col > self.dts.RasterXSize:
            col = np.nan
        row = int((xy[1] - gt[3]) / gt[5])
        if row < 0 or row > self.dts.RasterYSize:
            row = np.nan
        return (row, col)

    def getXYFromRowCol(self, rc: Tuple[int, int]) -> Tuple[float, float]:
        """Converts row,column to coodinate pair."""
        gt = self.dts.GetGeoTransform()
        x = gt[1] * rc[1] + gt[2] * rc[0] + gt[0]
        y = gt[4] * rc[1] + gt[5] * rc[0] + gt[3]
        return (x, y)

    def extractFromXY(self, xy: list) -> list:
        """Extract raster value at coordinate.

        Args:
            xy: is a list with a coordinate pair or a list of lists with coordinate pairs.

        """
        if isinstance(xy[0], numbers.Number) and len(xy) == 2:
            xy = [xy]
        result = []
        na = self.dts.GetRasterBand(1).GetNoDataValue()
        for pnt in xy:
            rc = self.getRowColFromXY(pnt)
            if not np.isnan(rc[0]) and not np.isnan(rc[1]):
                tmp = self.dts.ReadAsArray(
                    int(rc[1]), int(rc[0]), 1, 1).astype("float")
                tmp[tmp == na] = np.nan
                result.append(tmp.flatten().tolist())
            else:
                result.append([np.nan] * self.dts.RasterCount)
        return result

    def writeRaster(
        self, filename: str, driver: str = "GTiff", compression: bool = True
    ) -> None:
        """Writes the current raster to a files.

        Args:
            filename: a string with file name to be created.
            driver: GDAL driver to be used
            compression: if True, attempts to compress the raster in the file (depends on driver)

        """
        out_dt = gdal.GetDriverByName(driver)
        opt = []
        if compression:
            opt.append("COMPRESS=DEFLATE")
        if driver == "GTiff":
            opt.append("BIGTIFF=YES")
        out_dt.CreateCopy(filename, self.dts, 1, options=opt)
        out_dt = None
        self.source = filename

    def get_array(self,
        band: Optional[int] = None,
        convNA: bool = True) -> np.ndarray:
        """Returns the full array of the raster.

        Args:
            band: integer defining a band to get or None for all bands
            convNA: attempts to convert NA values to np.nan (fails for non float rasters)

        NOTE:
            Conversion to np.nan forces an attempt to convert array to float, so use with caution.
        """
        na = self.dts.GetRasterBand(1).GetNoDataValue()
        if band is None:
            arr = self.dts.ReadAsArray()
        else:
            arr = self.dts.GetRasterBand(band).ReadAsArray()
        if convNA:
            if not issubclass(arr.dtype.type, np.floating):
                arr = arr.astype("float")
            arr[arr == na] = np.nan
        return arr

    def get_subarray(self,
        rc: Tuple[int, int] = (0,0),
        blocksize: int = 250,
        band: Optional[int] = None,
        convNA: bool = True) -> np.ndarray:
        """Returns the full array of the raster.

        Args:
            rc: Corner position of the sub array to retrieve
            blocksize: side length of the subarray to retrieve.
            band: integer defining a band to get or None for all bands
            convNA: attempts to convert NA values to np.nan (fails for non float rasters)

        NOTE:
            Conversion to np.nan forces an attempt to convert array to float, so use with caution.
        """
        row = rc[0]
        col = rc[1]
        ys, xs = self.size
        xoff = blocksize
        if col + xoff > xs:
            xoff = xs - col
        yoff = blocksize
        if row + yoff > ys:
            yoff = ys - row

        na = self.dts.GetRasterBand(1).GetNoDataValue()
        if band is None:
            arr = self.dts.ReadAsArray(col, row, xoff, yoff)
        else:
            arr = self.dts.GetRasterBand(band).ReadAsArray(col, row, xoff, yoff)

        if convNA:
            if not issubclass(arr.dtype.type, np.floating):
                arr = arr.astype("float")
            arr[arr == na] = np.nan

        return arr

    def get_nodata_mask(self,
        band: Optional[int] = None,
        any: bool = True) -> np.ndarray:
        """ Returns the no data mask of the raster.

        Useful to avoid conversions to float when using np.nan.

        Args:
            band: either None (all bands) or an integer indicating band
            any: if True any band with NA defines mask, otherwise, all bands need to have NA

        """
        na = self.dts.GetRasterBand(1).GetNoDataValue()
        arr = self.get_array(band=band, convNA=False)
        mask = (arr == na).astype("bool")
        if band is None:
            mask = np.sum(mask, axis = 0)
            n = 0 if any else arr.shape[0]
            mask = (mask > n).astype("bool")
        return mask

    def block_iter(
        self,
        blocksize: int = 250,
        band: Optional[int] = None,
        read_arr: bool = True
    ) -> Iterator[Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]]:
        """Generator of array with raster values with specified size.

        Args:
            blocksize: the square size of the block, defining the shape of the yielded array (default is 250).
            band: Either 'None' for all bands or a specific raster band.
            read_arr: It allows to return the array (default) or just the
            position for each iterator block without attempt to read data.

        Yields:
            a list with location elements and the nnumpy array with data.
            First item in yielded list is a list with [row, column, current array, total number of arrays].

        """
        xs = self.dts.RasterXSize
        ys = self.dts.RasterYSize
        na = self.dts.GetRasterBand(1).GetNoDataValue()
        total = ceil(xs / blocksize) * ceil(ys / blocksize)
        i = 0
        arr = None
        for col in range(0, xs, blocksize):
            xoff = blocksize
            if col + xoff > xs:
                xoff = xs - col
            for row in range(0, ys, blocksize):
                yoff = blocksize
                if row + yoff > ys:
                    yoff = ys - row

                if read_arr:
                    if band is None:
                        arr = self.dts.ReadAsArray(col, row, xoff, yoff)
                    else:
                        arr = self.dts.GetRasterBand(band).ReadAsArray(col, row, xoff, yoff)
                    # Arrays are sent as float to accept np.nan
                    arr = arr.astype("float")
                    arr[arr == na] = np.nan
                i += 1
                yield ((row, col, i, total), arr)

    def set_array(
        self,
        arr: np.ndarray,
        rc: Tuple[int, int] = (0, 0),
        band: Optional[int] = None
    ) -> None:
        """Writes values of an array to raster.

        The array might represent the full raster, of it is a subset of it, the location given in [row, col] pair is needed.

        Args:
            arr: the array with values to be written.
            rc: the position of the array. If the full array, the position is the origin [0,0] (default).
            band: which band is to be written. If 'None' (default) it expects all bands in the array.

        """
        if band is None:
            self.dts.WriteArray(arr, rc[1], rc[0])
        else:
            self.dts.GetRasterBand(band).WriteArray(arr, rc[1], rc[0])
