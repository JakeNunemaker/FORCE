__author__ = ["Jake Nunemaker", "Matt Shields", "Philipp Beiter"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"
__status__ = "Development"


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
