#==================================================================#
# ANALYZE CRIME IN CHICAGO
#
# Cecile Murray
# CAPP 30254
#==================================================================#

import pandas as pd
import geopandas as gpd 
import censusdata as census

# TO DO
# pull down some ACS data about Chicago census tracts
# get shapefiles for Chicago census tracts
# point in poly merge
# answer summary stats questions


# pull down ACS data on race, income, unemployment, single-parent families, edu attainment
# tables B02001 (race), B19031 (median household income), B11001 (household type), S1501 (edu)

def get_census_data():

    hhold_vars = map(lambda x: 'B11001_00' + x + 'E', range(1, 10))
    race_vars = map(lambda x: 'B02001_00' + x + 'E', range(1, 9))
    all_vars = hhold_vars + race_vars


    data = census.download('acs5', 2017, all_vars)