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
        self,
        projects,
        filters={},
        regression_variables=[],
        status=["Installed"],
        drop_countries=[],
        drop_categorical=[],
        aggregate_countries={},
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
        status : list
            List of project statuses to consider.
            Default: ['Installed']
        drop_countries : list
            Countries to exclude from regression.
            Default: []
        drop_categorical : list
            Categorical variables to exclude from regression.
            Default: []
        aggregate_countries : dict
            Countries to aggregate to larger regions.
            Format: 'country': 'new region'
            Default: {}
        """

        self.regression_variables = list(
            set(
                [
                    "COD",
                    "Capacity MW (Max)",
                    "ProjectCost Currency",
                    "ProjectCost Mill",
                    *regression_variables,
                ]
            )
        )

        self._status = status
        self._drop_country = drop_countries
        self._drop_categorical = drop_categorical
        self._aggr = aggregate_countries

        self._data = self.clean_data(projects, self.regression_variables)
        self._processed = self.filter_and_process_data(self._data, filters)

    @property
    def raw_data(self):
        """Returns data before column filters are applied."""
        return self._data

    @property
    def processed_data(self):
        """Returns data after column filters are applied, a"""
        return self._processed

    def filter_and_process_data(self, data, filters):
        """
        Filters input `data` by `filters` and processes for regression analysis.

        Parameters
        ----------
        data : pd.DataFrame
        filters : dict
        """

        data = self.filter_data(data, filters)
        data = self.append_cumulative(data)
        data = self.process_data(data)
        return data

    def filter_data(self, data, filters):
        """
        Filters input `data` by any range filters in `filters` kwarg.

        Parameters
        ----------
        data : pd.DataFrame
        filters : dict
        """

        for col, filt in filters.items():
            try:
                data = self._filter_range(data, col, *filt)

            except KeyError as e:
                raise KeyError(f"Column name '{col}' not found.")

            except TypeError as e:
                try:
                    data[col] = data[col].astype(float)

                except ValueError:
                    raise TypeError(
                        f"Range filter not applicable for column '{col}'"
                    )

                data = self._filter_range(data, col, *filt)

        data = data.loc[data["Windfarm Status"].isin(self._status)]
        data = data.loc[~data["Country Name"].isin(self._drop_country)]

        if self._aggr:
            data["Country Name"] = data["Country Name"].apply(
                lambda x: self._aggr[x] if x in self._aggr.keys() else "Other"
            )

        else:
            data["Country Name"] = "Global"

        return data

    def append_cumulative(self, data):
        """
        Append cumulative capacity to input `data`.

        Parameters
        ----------
        data : pd.DataFrame
        """

        ret = data.copy()  # .reset_index(drop=True) #.sort_values("COD").

        # cumulative = dict(zip(ret["COD"], ret["Capacity MW (Max)"].cumsum(axis=0)))
        yearly = ret.groupby(["COD"]).sum()["Capacity MW (Max)"]
        cumulative = dict(zip(yearly.index, yearly.cumsum(axis=0)))
        ret["Cumulative Capacity"] = ret["COD"].apply(lambda x: cumulative[x])

        return ret

    def process_data(self, data):
        """
        Appends categorical columns to `data`.

        Parameters
        ----------
        data : pd.DataFrame
        """

        countries = data["Country Name"].unique()
        for c in countries:
            if c not in self._drop_categorical:
                data[c] = data["Country Name"].apply(
                    lambda x: 1 if x == c else 0
                )
                self.regression_variables.append(c)

        self.regression_variables.remove("Country Name")

        return data

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

        return data.loc[(data[col] > min) & (data[col] < max)]

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
