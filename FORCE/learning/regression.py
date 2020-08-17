__author__ = ["Matt Shields", "Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"
__status__ = "Development"


import numpy as np
import pandas as pd

from FORCE.library import ex_rates


class RegressionData:
    def __init__(
        self, projects, filters={}, regression_variables=[], **kwargs
    ):
        """
        Creates an instance of `RegressionData`.

        Parameters
        ----------
        projects : pd.DataFrame
            Project dataset.
        filters : dict
            Numeric filters on columns.
            Format: 'col': (min, max (optional))
        regression_variables : list
            List of variables used in the regression. Dataset will be filtered
            so that all datapoints have data in these columns.
        """

        required_columns = [
            "COD",
            "Capacity MW (Max)",
            "ProjectCost Currency",
            "ProjectCost Mill",
            *regression_variables,
        ]

        self._data = self.clean_data(projects, required_columns)
        self._filtered = self.filter_data(self._data, filters)

    @property
    def raw_data(self):
        """Returns data before column filters are applied."""
        return self.append_cumulative(self._data)

    @property
    def filtered_data(self):
        """Returns data after column filters are applied."""
        return self.append_cumulative(self._filtered).reset_index(drop=True)

    @staticmethod
    def append_cumulative(data):
        """
        Append cumulative capacity to input `data`.
        
        Parameters
        ----------
        data : pd.DataFrame
        """
        ret = data.copy()
        ret["Cumulative Capacity"] = data["Capacity MW (Max)"].cumsum()
        return ret

    @classmethod
    def clean_data(cls, data, required_columns):
        """
        Removes entries that don't have data in `required_columns`, converts
        currencies to USD and calculates CAPEX per kW.

        Parameters
        ----------
        data : pd.DataFrame
        required_columns : list
        """

        data = data.loc[~data[required_columns].isnull().any(axis=1)].copy()
        data["CAPEX_conv"] = data.apply(
            cls.conv_currency,
            axis=1,
            id_col="ProjectCost Mill",
            val_col="ProjectCost Currency",
        )
        data["CAPEX_per_kw"] = (data["CAPEX_conv"] * 1e6) / (
            data["Capacity MW (Max)"] * 1e3
        )

        return data

    @classmethod
    def filter_data(cls, data, filters):
        """
        Filters input `data` by any range filters in `filters` kwarg.

        Parameters
        ----------
        data : pd.DataFrame
        filters : dict
        """

        for col, filt in filters.items():
            try:
                data = cls._filter_range(data, col, *filt)

            except KeyError as e:
                raise KeyError(f"Column name '{col}' not found.")

            except TypeError as e:
                try:
                    data[col] = data[col].astype(float)

                except ValueError:
                    raise TypeError(
                        f"Range filter not applicable for column '{col}'"
                    )

                data = cls._filter_range(data, col, *filt)

        return data

    @staticmethod
    def _filter_range(data, col, min, max=np.inf):
        """
        Filters input `data` by values in `col` where `min` <= val <= `max`.

        Parameters
        ----------
        data : pd.DataFrame
        col : str
        min : int | float
        max : int | float
            Default: np.inf
        """

        return data.loc[(data[col] >= min) & (data[col] <= max)]

    @staticmethod
    def conv_currency(row, id_col, val_col, output="USD"):
        """
        Converts currency of `val_col` based on `id_col`.

        Parameters
        ----------
        row : pd.Series
        id_col : str
            Input currency identifier column.
        val_col : str
            Input value column.
        output : str
            Output currency identifier.
            Default: 'USD'
        """

        year = pd.to_datetime(row["Financial Close"]).year
        if np.isnan(year):
            year = row["COD"] - 2

        currency = row[id_col]

        # TODO: Update exchange rates and remove this.
        if year > 2017.0:
            year = 2017.0
        elif year < 1990.0:
            year = 1990.0

        ex_rate = ex_rates[currency][year]

        if output != "USD":
            return NotImplemented(
                "Currency conversion other than 'USD' not supported yet."
            )

        return row[val_col] / ex_rate
