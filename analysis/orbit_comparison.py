__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os
import numpy as np
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
from ORBIT import ProjectManager, load_config
from ORBIT.core.library import initialize_library
from FORCE.learning import Regression
from plot_routines import scatter_plot, plot_forecast


DIR = os.path.split(__file__)[0]
LIBRARY = os.path.join(DIR, "library")
initialize_library(LIBRARY)


# TODO: Reindex all data to the same starting year before anything happens.
# TODO: May need to revise how forecasts are input


### Initialize Data
## Forecast
FORECAST_FP = os.path.join(DIR, "data", "2021_forecast.csv")
FORECAST = pd.read_csv(FORECAST_FP).set_index("year").to_dict()["capacity"]

## Regression Settings
PROJECTS = pd.read_csv(os.path.join(DIR, "data", "2021_OWMR.csv"), header=2)
FILTERS = {
    'Capacity MW (Max)': (149, ),
    'Full Commissioning': (2014, 2021),
}
TO_AGGREGATE = {
    'United Kingdom': 'United Kingdom',
    'Germany': 'Germany',
    'Netherlands': 'Netherlands',
    'Belgium' : 'Belgium',
    'China': 'China',
    'Denmark': 'Denmark',
}
TO_DROP = []
PREDICTORS = [
            'Country Name',
            'Water Depth Max (m)',
            # 'Turbine MW (Max)',
            'Capacity MW (Max)',
            'Distance From Shore Auto (km)',
            ]


## ORBIT Sites + Configs
ORBIT_SITES = {
    "Site 1": {
        2021: "site_1_2021.yaml",
        2025: "site_1_2025.yaml",
        2030: "site_1_2030.yaml",
        2035: "site_1_2035.yaml"
    },

    "Site 2": {
        2021: "site_2_2021.yaml",
        2025: "site_2_2025.yaml",
        2030: "site_2_2030.yaml",
        2035: "site_2_2035.yaml"
    }
}


### Functions
def run_regression(projects, filters, to_aggregate, to_drop, predictors):
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
        regression_variables=predictors,
        aggregate_countries=to_aggregate,
        drop_categorical=["United Kingdom"],
        drop_country=to_drop,
        log_vars=['Cumulative Capacity', 'CAPEX_per_kw'],
    )
    print(regression.summary)
    return regression

def stats_check(regression):
    summary_stats = {'R2': regression.r2,
                     'Adjusted R2': regression.r2_adj,
                     'Experience factor': regression.cumulative_capacity_fit,
                     'Experience factor standard error': regression.cumulative_capacity_bse,
                     'Learning rate': regression.learning_rate,
                     }
    predictor_stats = zip(regression.params_dict.values(),
                          regression.pvalues.keys(),
                          regression.pvalues.values,
                          regression.vif)

    # Write stats results to Excel
    xlsfile = "results/statistics/stats_output.xlsx"
    workbook = xlsxwriter.Workbook(xlsfile)
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    # Write all data to workbook.  Start with scalars then per-predictor values
    for k, v in summary_stats.items():
        worksheet.write(row, col, k)
        worksheet.write(row, col+1, v)
        row+=1
    row+=1
    worksheet.write(row, col, 'Predictor variable')
    worksheet.write(row, col + 1, 'Coefficient')
    worksheet.write(row, col+2, 'P-value')
    worksheet.write(row, col+3, 'VIF')
    row+=1
    for b, var, p, v in predictor_stats:
        worksheet.write(row, col, var)
        worksheet.write(row, col + 1, b)
        worksheet.write(row, col+2, p)
        worksheet.write(row, col+3, v)
        row+=1
    workbook.close()

    # Plot residuals
    res_x = regression.fittedvalues
    res_y = regression.residuals

    return res_x, res_y


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

    orbit_outputs = []
    for name, configs in sites.items():

        site_data = pd.DataFrame(index=years)

        for yr, c in configs.items():

            config = load_config(os.path.join(DIR, "orbit_configs", "fixed", c))
            weather_file = config.pop("weather", None)

            if weather_file is not None:
                weather = pd.read_csv(os.path.join(DIR, "library", "weather", weather_file)).set_index("datetime")

            else:
                weather = None

            #TODO: better indexing
            if yr == 2021:
                ncf_i = config['project_parameters']['ncf']
                opex_i = config['project_parameters']['opex']
                fcr_i = config['project_parameters']['fcr']
            elif yr == 2035:
                ncf_f = config['project_parameters']['ncf']
                opex_f = config['project_parameters']['opex']
                fcr_f = config['project_parameters']['fcr']

            project = ProjectManager(config, weather)
            project.run()

            site_data.loc[int(yr), "ORBIT"] = project.total_capex_per_kw

        min_yr = min(configs.keys())  # TODO: What if min_yr doesn't line up with first forecast year?
        c = site_data.loc[min_yr, "ORBIT"] / (regression.installed_capacity ** b0)
        site_data.loc[min_yr, "Regression"] = c * upcoming[yr] ** b0
        for yr in years[1:]:
            site_data.loc[yr, "Regression"] = c * upcoming[yr] ** b0

        # Define Opex, NCF, FCR arrays
        OPEX = (opex_i, opex_f)
        NCF = (ncf_i, ncf_f)
        FCR = (fcr_i, fcr_f)
        opex = {yr: val for yr, val in zip(years, np.linspace(*OPEX, len(years)))}
        ncf = {yr: val for yr, val in zip(years, np.linspace(*NCF, len(years)))}
        fcr = {yr: val for yr, val in zip(years, np.linspace(*FCR, len(years)))}

        site_data["OpEx"] = opex.values()
        aep = {k: v * 8760 for k, v in ncf.items()}  # MWh
        site_data["AEP"] = aep.values()
        site_data["FCR"] = fcr.values()
        site_data["LCOE"] = 1000 * (site_data["FCR"] * site_data["Regression"] + site_data["OpEx"]) / site_data["AEP"]
        site_data["Site"] = name

        orbit_outputs.append(site_data)

    combined_outputs = pd.concat(orbit_outputs)

    return combined_outputs


### Main Script
if __name__ == "__main__":
    
    # Forecast
    years, linear_forecast = linearize_forecast(FORECAST)

    # Regression
    regression = run_regression(PROJECTS, FILTERS, TO_AGGREGATE, TO_DROP, PREDICTORS)
    res_x, res_y = stats_check(regression)
    b0 = regression.cumulative_capacity_fit
    upcoming_capacity = {
        k: v - regression.installed_capacity for k, v in linear_forecast.items()
    }
    # ORBIT Results
    combined_outputs = run_orbit_configs(ORBIT_SITES, b0, upcoming_capacity, years)
    avg_start = pd.pivot_table(combined_outputs.reset_index(), values='ORBIT', index='index').iloc[0].values[0]
    std_start = pd.pivot_table(combined_outputs.reset_index(), values='ORBIT', index='index', aggfunc=np.std).iloc[0].values[0]

    ### Plotting
    # Forecast
    plot_forecast(
        regression.installed_capacity,
        avg_start,
        std_start,
        b0,
        upcoming_capacity,
        regression.cumulative_capacity_bse,
        # data_file='results/data.csv',
        fname='results/forecast.png'
    )
    # Residuals
    scatter_plot(res_x, res_y, 'Fitted values (log of CapEx)', 'Residuals', fname='results/statistics/residuals.png')

    # TODO:
    #   1. Line up x ticks in plot_forecast
    #   3. Plots for high/medium/low deployment projectsions
