__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os
import numpy as np
import pandas as pd
from ORBIT import ProjectManager, load_config
from ORBIT.core.library import initialize_library
from FORCE.learning import Regression


DIR = os.path.split(__file__)[0]
LIBRARY = os.path.join(DIR, "library")
initialize_library(LIBRARY)


# TODO: Reindex all data to the same starting year before anything happens.
# TODO: May need to revise how forecasts are input


### Initialize Data
## Forecast
FORECAST_FP = os.path.join(DIR, "data", "2021_forecast.csv")
FORECAST = pd.read_csv(FORECAST_FP).set_index("year").to_dict()["capacity"]

## LCOE Parameters
OPEX = (129, 65)       # (start of forecast, end of forecast)
NCF = (0.486, 0.516)   # (start of forecast, end of forecast)
FCR = (0.1, 0.1)       # (start of forecast, end of forecast)


## Regression Settings
PROJECTS = pd.read_csv(os.path.join(DIR, "data", "2021_OWMR.csv"), header=2)
FILTERS = {
    'Capacity MW (Max)': (149, ),
    'Full Commissioning': (2014, 2021),
    'CAPEX_per_kw': (800, 8000.0)
}
TO_AGGREGATE = {
    'United Kingdom': 'United Kingdom',
    'Germany': 'Germany',
    'Netherlands': 'Netherlands',
    'Belgium' : 'Belgium',
    'China': 'China',
}
TO_DROP = []


## ORBIT Sites + Configs
ORBIT_SITES = {
    "Site 1": {
        2021: "site_1_2020.yaml",
        2025: "site_1_2025.yaml",
        2030: "site_1_2020.yaml"
    },

    "Site 2": {
        2021: "site_2_2020.yaml",
        2025: "site_2_2025.yaml",
        2030: "site_2_2020.yaml"
    }
}


### Functions
def run_regression(projects, filters, to_aggregate, to_drop):
    """
    Run FORCE Regression with given settings.

    Parameters
    ----------
    projects : DataFrame
    filters : dict
    to_aggregate : dict
    to_drop : list
        List of countries to drop.
    """

    regression = Regression(
        projects,
        y_var="log CAPEX_per_kw",
        filters=filters,
        regression_variables=[
            'Country Name',
            'Water Depth Max (m)',
            'Turbine MW (Max)', 
            'Capacity MW (Max)',
            'Distance From Shore Auto (km)'
            ],
        aggregate_countries=to_aggregate,
        drop_categorical=["United Kingdom"],
        drop_country=to_drop,
        log_vars=['Cumulative Capacity', 'CAPEX_per_kw']
    )

    return regression


def linearize_forecast(forecast):
    """
    Linearize the forecasted capacity over forecast period.

    Parameters
    ----------
    forecast : dict
    """

    years = np.arange(min(forecast.keys()), max(forecast.keys()) + 1)
    linear = np.linspace(min(forecast.values()), max(forecast.values()), len(years))
    f2 = {k: linear[i] for i, k in enumerate(years)}

    return years, f2


def _zip_into_years(start, stop, years):
    return {yr: val for yr, val in zip(years, np.linspace(start, stop, len(years)))}


def run_orbit_configs(sites, b0, upcoming, years):
    """"""

    opex = {yr: val for yr, val in zip(years, np.linspace(*OPEX, len(years)))}
    ncf = {yr: val for yr, val in zip(years, np.linspace(*NCF, len(years)))}
    fcr = {yr: val for yr, val in zip(years, np.linspace(*FCR, len(years)))}

    orbit_outputs = []
    for name, configs in sites.items():

        site_data = pd.DataFrame(index=years)

        for yr, c in configs.items():

            config = load_config(os.path.join(DIR, "orbit_configs", "fixed", c))
            weather_file = config.pop("weather", None)
            turbine = config.get("turbine")
            turbine_rating = float(turbine.split("MW")[0])  # TODO: Revise. Is there a numeric turbine rating property in ProjectManager?

            if weather_file is not None:
                weather = pd.read_csv(os.path.join(DIR, "library", "weather", weather_file)).set_index("datetime")

            else:
                weather = None

            project = ProjectManager(config, weather)
            project.run()

            site_data.loc[int(yr), "ORBIT"] = project.total_capex_per_kw

        min_yr = min(configs.keys())  # TODO: What if min_yr doesn't line up with first forecast year?
        c = site_data.loc[min_yr, "ORBIT"] / (regression.installed_capacity ** b0)
        site_data.loc[min_yr, "Regression"] = c * upcoming[yr] ** b0
        for yr in years[1:]:
            site_data.loc[yr, "Regression"] = c * upcoming[yr] ** b0

        site_data["OpEx"] = opex.values()
        aep = {k: v * turbine_rating * 8760 for k, v in ncf.items()}  # MWh
        site_data["AEP"] = aep.values()
        site_data["FCR"] = fcr.values()
        site_data["LCOE"] = (site_data["FCR"] * site_data["Regression"] / 1000 + site_data["OpEx"]) / site_data["AEP"]
        # TODO: check units for above                     $/MW                        $/MW/year                MWh
        site_data["Site"] = name

        orbit_outputs.append(site_data)

    combined_outputs = pd.concat(orbit_outputs)

    return combined_outputs


### Main Script
if __name__ == "__main__":
    
    # Forecast
    years, linear_forecast = linearize_forecast(FORECAST)

    # Regression
    regression = run_regression(PROJECTS, FILTERS, TO_AGGREGATE, TO_DROP)
    b0 = regression.cumulative_capacity_fit
    upcoming_capacity = {
        k: v - regression.installed_capacity for k, v in linear_forecast.items()
    }

    # ORBIT Results
    combined_outputs = run_orbit_configs(ORBIT_SITES, b0, upcoming_capacity, years)
    print(combined_outputs)

    # output_std = combined_outputs.groupby([combined_outputs.index]).std()
    # Plotting
    # TODO: 
