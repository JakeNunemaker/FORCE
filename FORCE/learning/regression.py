__author__ = ["Matt Shields", "Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"
__status__ = "Development"


import numpy as np
import pandas as pd
import statsmodels.api as sm

from FORCE.library import ex_rates


class Regression:
    def __init__(
        self,
        projects,
        y_var,
        filters={},
        regression_variables=[],
        status=["Installed"],
        drop_countries=[],
        drop_categorical=[],
        aggregate_countries={},
        log_vars=[],
        add_vars={},
        **kwargs,
    ):
        """
        Creates an instance of `Regression`.

        Parameters
        ----------
        projects : pd.DataFrame
            Project dataset.
        y_var : str
            Y variable for the linear regression model. Should either match an
            input column or a calculated log column, 'log {column}'.
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
        log_vars : list
            Columns to take log of.
        add_vars : dict
            Additional explanatory data.
            Format: {'key': {2005: val1, 2006: val2, ...}}
        """

        self.regression_variables = regression_variables

        self._y = y_var
        self._status = status
        self._drop_country = drop_countries
        self._drop_categorical = drop_categorical
        self._aggr = aggregate_countries
        self._log = log_vars
        self._add = add_vars

        self._data = self.clean_data(
            projects,
            [
                "COD",
                "Capacity MW (Max)",
                "ProjectCost Mill",
                "ProjectCost Currency",
                *self.regression_variables,
            ],
        )
        self._processed = self.filter_and_process_data(self._data, filters)

        self.multi_linear_regression(**kwargs)

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
        data = self.process_categorical_data(data)
        data = self.preprocess_data(data)
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

        ret = data.copy().sort_values("COD")
        yearly = ret.groupby(["COD"]).sum()["Capacity MW (Max)"]
        cumulative = dict(zip(yearly.index, yearly.cumsum(axis=0)))
        ret["Cumulative Capacity"] = ret["COD"].apply(lambda y: cumulative[y])
        self.regression_variables.append("Cumulative Capacity")

        return ret

    def process_categorical_data(self, data):
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

    def preprocess_data(self, data):
        """
        Preprocess data for regression. Takes log of any columns in `self._log`,
        and appends any additional data in `self._add`.

        Parameters
        ----------
        data : pd.DataFrame
        """

        for item in self._log:
            log_item = f"log {item}"
            data[log_item] = data[item].apply(np.log)

            if item in self.regression_variables:
                self.regression_variables.remove(item)
                self.regression_variables.append(log_item)

        for key, val in self._add.items():
            projects[key] = projects["COD"].apply(lambda y: val[y])
            self.regression_variables.append(key)

        return data

    def clean_data(self, data, required_columns):
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
            self.conv_currency,
            axis=1,
            id_col="ProjectCost Mill",
            val_col="ProjectCost Currency",
        )
        data["CAPEX_per_kw"] = (data["CAPEX_conv"] * 1e6) / (
            data["Capacity MW (Max)"] * 1e3
        )

        return data

    def multi_linear_regression(self, **kwargs):
        """
        Conduct multivariate linear regression controlling for desired variables.
        """

        data = self.processed_data.copy()

        X = data[self.regression_variables]
        Y = data[self._y]
        X2 = sm.add_constant(X)

        if len(self.regression_variables) > 1:
            self.vif = self.calculate_vif(X2)

        sm_regressor = sm.OLS(Y, X2).fit()
        print(sm_regressor.summary())

        self.summary = sm_regressor.summary()
        self.r2 = sm_regressor.rsquared
        self.params = sm_regressor.params
        self.params_dict = dict(self.params)
        self.bse_dict = dict(sm_regressor.bse)

    def calculate_vif(self, df):
        """
        Calculate variance inflation factor for all columns in `df`.

        Parameters
        ----------
        df : pd.DataFrame
        """

        vif = []
        for name, data in df.iteritems():
            r_sq_i = sm.OLS(data, df.drop(name, axis=1)).fit().rsquared
            vif.append(1.0 / (1.0 - r_sq_i))

        print(f"Variance Inflation Factor:")
        print(pd.DataFrame(list(zip(df.columns, vif))))
        print("\n")

        return vif

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
