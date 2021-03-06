__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"
__status__ = "Development"


import os

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


_DIR = os.path.split(os.path.abspath(__file__))[0]


with open(os.path.join(_DIR, "exchange_rates.yaml"), "r+") as f:
    ex_rates = yaml.load(f, Loader=Loader)
