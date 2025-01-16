import numpy as np
import numbers
from spartann.spatial import Raster
from typing import Tuple


def str2num(s: str) -> int | float | str:
    """Attempts to convert a string to integer or float.

    "12" is converter to integer while "12." is converted to float.
    If not possible, returns the original string.

    Args:
        s: string to convert.
    """
    if isinstance(s, str):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s


class DataTable(object):
    """ "The DataTable provides interface for a tabular point data for supervised learning.
    It provides methods to read point coodinates, classes and to read/add predictor data.
    Methods provided allow interaction with Raster object to extract data.
    """

    def __init__(self, crd: list, classes: list, class_names: list|None = None) -> None:
        """Initialization of DataTable instance.

        Note: lists are converted to numpy arrays.

        Args:
            crd: a list of coordinate pairs in the for [[X1,Y1], [X2,Y2], ...]
            classes: a list of classes to be used in the supervised learning. Can be multiples classes for multiple outputs as [[cl1_1,cl2_1,...], [cl1_2, cl2_2,...]].
            class_names: a list as same length as 'classes' with classes names. If 'None', than classes will have names such ["Class1", "Class2", ...]

        """
        if not isinstance(crd, list):
            raise Exception("Coordinates must be a list")
        if isinstance(crd[0], numbers.Number) and len(crd) == 2:
            crd = [crd]
        n = len(crd)
        if len(classes) != n:
            raise Exception("'crd' and 'classes' must have the same length.")
        if not class_names:
            class_names = ["Class"+str(x) for x in range(len(classes[0]))]
        elif len(class_names) != len(classes[0]):
            raise Exception("Names for classes must have same length as provided classes.")
        self.points = np.array(crd)
        self.classes = np.array(classes)
        self.class_names = [str(x) for x in class_names]
        self.data = None
        self.datacolnames = []
        self.scaling = None

    def __len__(self) -> int:
        """Returns the number of data points available."""
        return self.points.shape[0]

    @property
    def ndatacols(self) -> int:
        """Returns the number of data columns available.

        Note: only data columns (no coordinates or classes are considered)
        """
        if self.data is None:
            return 0
        return self.data.shape[1]

    @property
    def nclasses(self) -> int:
        """Returns the number of classes columns available.

        Note: only classes columns (no coordinates or data are considered)
        """
        return self.classes.shape[1]

    @property
    def has_nan(self) -> bool:
        """Checks if there is *any* missing data in the data table."""
        if self.data is None:
            return False
        return bool(np.any(np.isnan(self.data)))

    @property
    def has_row_nan(self) -> bool:
        """Check if there are rows with missing data only."""
        if self.data is None:
            return False
        return bool(np.any(np.all(np.isnan(self.data), axis=1)))

    def where_nan(self, any: bool = False) -> list | None :
        """Get rows indices where missing data is found.

        Args:
            any: A boolean indicating if rows with *any* missing data are to be retrieved or only rows with *full* missing data (all data columns are missing data).
        """
        if self.data is None:
            return None
        if any:
            rows = np.where(np.any(np.isnan(self.data), axis=1))[0]
        else:
            rows = np.where(np.all(np.isnan(self.data), axis=1))[0]
        return rows.tolist()

    @property
    def is_scaled(self) -> bool:
        """Checks if data is scaled to Z-scores."""
        if self.scaling is None:
            return False
        return True

    def __str__(self) -> str:
        """Returns the string representation of the DataTable instance."""
        s = "X|Y|" + "|".join(self.class_names)
        if self.ndatacols > 0:
            s += "|" + "|".join(self.datacolnames)
        s += "\n"
        for i in range(len(self)):
            pnt = "|".join([str(x) for x in self.points[i,]])
            cl = "|".join([str(x) for x in self.classes[i]])
            s += f"{pnt}|{cl}"
            if self.ndatacols > 0 and self.data is not None:
                s += "|" + "|".join([str(x) for x in self.data[i,]])
            s += "\n"
        return s

    def write_to_file(self, filename):
        """ Export current data to a file."""
        with open(filename, "w") as stream:
            stream.write(str(self))

    @classmethod
    def from_file(cls, filename: str, sep: str = ";"):
        """Inititates a DataTable instance from a file.

        The file is expected to be structured with Lon;Lat;Class with or without extra columns for further data.
        The class given, if numeric, will be considered as a single class. If the class is a string, then the number of classes added is the number of unique strings in the column.

        Args:
            filename: the path and name for a text file with data.
            sep: The separator symbol used in the text file.

        """
        pnts = []
        cl = []
        dt = []
        classname = None
        datacolnames = []
        with open(filename, "r") as stream:
            for line in stream:
                line = line.strip().split(sep)
                if classname:
                    pnts.append([float(x) for x in line[:2]])
                    cl.append([str2num(line[2])])
                    if len(line) > 3:
                        dt.append([float(x) for x in line[3:]])
                else:
                    classname = line[2:3]
                    if len(line) > 3:
                        datacolnames = line[3:]

        if type(cl[0][0]) is str:
            # The format implies tha strings in class names are for multi class objects.
            classname = sorted(list(set([x[0] for x in cl])))
            cl = [[1 if x[0] == cl else 0 for cl in classname] for x in cl]
        dtab = cls(pnts, cl, classname)
        if len(dt) > 0:
            dt = [list(x) for x in zip(*dt)]
            for i in range(len(dt)):
                dtab.add_datacolumn(dt[i], datacolnames[i])
        return dtab

    def addpoint(self, pnt: Tuple[float, float], cl: list) -> None:
        """Adds a point with respective classification.

        Args:
            pnt: list with coordinate pair
            cl: list with classification value for the classes available

        """
        if len(pnt) != 2:
            raise Exception("Point should be a coordinate pair.")
        if len(cl) != self.nclasses:
            raise Exception(f"Classes for the point should have {self.nclasses} values.")

        self.points = np.row_stack((self.points, pnt))
        self.classes = np.append(self.classes, cl)
        if self.data:
            dt = np.repeat(np.nan, self.data.shape[1])
            self.data = np.vstack((self.data, dt))

    def rmpoint(self, i: int) -> None:
        """Removes a points from the data table.

        Args:
            i: the line to be removed.
        """
        self.points = np.delete(self.points, i, 0)
        self.classes = np.delete(self.classes, i)
        if self.data:
            self.data = np.delete(self.data, i, 0)

    def add_datacolumn(self, dt: list, name: str|None = None) -> None:
        """Adds a data column to available data points.

        Args:
            dt: a list of values to be added. It needs to have the same length as data points available.
            name: a name given to the data column. If 'None' than colnames are fiven by default as "Col1".

        """
        n = len(self)
        if len(dt) != n:
            raise ValueError(
                f"Data must have same length as available points ({n})."
            )
        if not name:
            name = "Col" + str(len(self.datacolnames)+1)

        # Check if names are not duplicates
        if name in self.datacolnames:
            raise ValueError(
                f"The name '{name}' already exists."
            )

        if self.data is None:
            self.data = np.array(dt)
            self.data.reshape(len(self.data), 1)
            self.datacolnames = [name]
        else:
            self.data = np.column_stack((self.data, dt))
            self.datacolnames.append(name)

    def add_datacolumns(self, dt: list, names: list|None = None) -> None:
        """Adds multiple data columns to available data points.

        Args:
            dt: a list of lists with values to be added. It needs to have the same length as data points available. Main list represent lines and inside list represent column values to be added.
            names: a list with length = len(dt), with data columns names. Note: if not given, default names will be ["Col1", "Col2", ...].
        """

        if not isinstance(dt[0], list):
            raise ValueError(
                "The values provided are not organized in list of lists (lines and columns)"
            )
        if names and len(names) != len(dt):
            raise ValueError(
                "The column names must have same length as data columns provided."
            )
        dt = np.array(dt)
        for col in range(dt.shape[1]):
            if names:
                self.add_datacolumn(dt[:,col], names[col])
            else:
                self.add_datacolumn(dt[:,col])

    def rm_datacolumn(self, i: int) -> None:
        """ Removes a column from data table.

          Note: it ignores coordinates and classes. Only data is considered."""
        self.data = np.delete(self.data, i, 1)
        _ = self.datacolnames.pop(i)

    def rm_data(self) -> None:
        """Cleans all data table.
         Note: it does not erase coordinates neither classes."""
        self.data = None
        self.datacolnames = []

    def setDataRow(self, row: int, data: list) -> None:
        """TODO: ADDA DESCRPT"""
        if len(data) == self.ndatacols:
            self.data[row,:] = data
        else:
            raise ValueError("Provided data has different number of items than columns available.")


    def getDataFromRaster(self, rst: Raster, overwrite: bool = False) -> None:
        """ Imports data from a Raster instance at the coordinate point in datatable.

        If there is no data available, it adds as many data columns as bands in raster. If data is available,
        it either overwrites current data or tries to fill missing data only.

        Args:
            rst: a Raster object instance with data to be extracted. Must be in same coordinate system and cover the extent of data points.
            overwrite: a bolean indicating if all data is to be overwritten with new raster data or only missing data. Note: it onlys consideres rows where all columns have missing data.

        """
        nc = self.ndatacols
        points = self.points
        dt = rst.extractFromXY(points.tolist())
        names = rst.bandnames
        if nc == 0:
            dt = [list(x) for x in zip(*dt)]
            for i in range(len(dt)):
                self.add_datacolumn(dt[i], names[i])
        else:
            if rst.nbands != nc:
                raise ValueError("Number of raster bands do not correspond to number of available columns.")
            if overwrite:
                rows = range(len(self))
            else:
                if not self.has_row_nan:
                    print("No missing data found. Consider setting 'overwrite' to True.")
                    return None
                rows = np.where(np.all(np.isnan(self.data), axis=1))[0]
            for row in rows:
                self.setDataRow(row, dt[row])

    def exclude_nan(self):
        """ Excludes all points that have *any* missing data (np.nan) in the data table.

        It excludes, data, coordinates and classes.

        """
        if self.has_nan:
            mask = np.invert(np.any(np.isnan(self.data), 1))
            self.points = self.points[mask,]
            self.classes = self.classes[mask,]
            self.data = self.data[mask,]
            print(f"{sum(np.invert(mask))} rows excluded.")

    def getRow(self, row:int) -> list:
        """ Returns row point, class and data.

        Args:
            row: an integer referring to row to retrieve (starts at 0)
        """
        if row < len(self):
            point = self.points[row,:]
            classp = self.classes[row]
            data = None
            if self.data is not None:
                data = self.data[row:]
            return [point, classp, data]
        else:
            raise IndexError("Row index out of bounds.")

    @property
    def getData(self) -> np.ndarray:
        """ Returns all data."""
        return self.data

    @property
    def getPoints(self) -> np.ndarray:
        """Returns the coordinate table."""
        return self.points

    @property
    def getClasses(self) -> np.ndarray:
        """Returns the class vector."""
        return self.classes

    @property
    def means(self) -> np.ndarray:
        """Returns an array of the current mean of data columns."""
        return self.data.mean(0)

    @property
    def sdevs(self) -> np.ndarray:
        """Returns an array of the current standard deviation of data columns."""
        return self.data.std(0)

    @property
    def scale_means(self) -> list[float]|None:
        """If data is scaled, returns the original means vector of the data."""
        if self.is_scaled:
            return self.scaling[0]
        return None

    @property
    def scale_sdevs(self) -> list[float]|None:
        """If data is scaled, returns the original standard deviation vector of the data."""
        if self.is_scaled:
            return self.scaling[1]
        return None

    def scaleData(self) -> None:
        """Scales data to Z-scores if is not scaled."""
        if not self.is_scaled:
            self.scaling = [self.means, self.sdevs]
            self.data = (self.data - self.means) / self.sdevs
