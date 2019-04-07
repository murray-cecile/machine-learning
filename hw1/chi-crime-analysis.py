#==================================================================#
# ANALYZE CRIME IN CHICAGO
#
# Cecile Murray
# CAPP 30254
#==================================================================#

import pandas as pd
import geopandas as gpd 
import censusdata as census
from shapely.geometry import Point


TRACTCODE = 2
BLKGRPCODE = 3
HSBEGIN = 19
HSEND = 34
BABEGIN = 39
EDUEND = 42


# TO DO
# figure out why some of the education variables are NaNs
# point in poly merge
# answer summary stats questions




def get_census_data():
    ''' pull down ACS data for Cook County IL on race, income, unemployment, single-parent families, edu attainment
      tables B02001 (race), B19013 (median household income), B11001 (household type), B15003 (edu) '''

    # construct lists of variables for the API query
    hhold_vars = list(map(lambda x: 'B11001_00' + str(x) + 'E', range(1, 10)))
    race_vars = list(map(lambda x: 'B03002_' + str(x).zfill(3) + 'E', range(1, 21)))
    edu_vars = list(map(lambda x: 'B15003_' + str(x).zfill(3) + 'E', range(1, 26)))
    

    # make the query for all block groups in Cook County, in two chunks because the API only takes 
    data = census.download('acs5', 2017,
        census.censusgeo([('state', '17'), ('county', '031'), ('block group', '*')]),
        hhold_vars + race_vars).reset_index()
    edu = census.download('acs5', 2017,
        census.censusgeo([('state', '17'), ('county', '031'), ('block group', '*')]),
        edu_vars + ['B19013_001E']).reset_index()

    data = pd.merge(data, edu, on='index')
    # print(data.head())

    # create unique FIPS ID
    data['bg'] = data['index'].apply(lambda x: '17031' + x.geo[TRACTCODE][1] + x.geo[BLKGRPCODE][1])
    
    # compute variables of interest
    data['pct_1parent'] = data['B11001_004E'] / data['B11001_001E']
    data['pct_alone'] = data['B11001_008E'] / data['B11001_001E']
    data['pct_white'] = data['B03002_003E'] / data['B03002_001E']
    data['pct_black'] = data['B03002_004E'] / data['B03002_001E']
    data['pct_hisp'] = data['B03002_012E'] / data['B03002_001E']
    data['pct_nohs'] = data.iloc[HSBEGIN:HSEND].sum(axis=1) / data['B15003_001E']
    data['pct_BA'] = data.iloc[BABEGIN:EDUEND].sum(axis=1) / data['B15003_001E']
    data['medinc'] = data['B19013_001E']

    rv = data[['bg', 'pct_1parent', 'pct_alone', 'pct_white', 'pct_black', 'pct_hisp', 'B15003_001E', 'pct_nohs', 'pct_BA', 'medinc']]
    print(rv.head())

    return rv


def prep_census_shp():
    ''' prep the block group shapefile for merges with other data'''

    cook_blks = gpd.read_file('geo/Cook_bg.geojson')
    cook_blks['bg'] = cook_blks['STATEFP'] + cook_blks['COUNTYFP'] + cook_blks['TRACTCE'] + cook_blks['BLKGRPCE']

    return cook_blks


def load_crime_data():
    '''read in crime data for 2017 and 2018, convert points to Point'''

    crime17 = pd.read_csv('raw/Crimes-2017.csv')
    crime18 = pd.read_csv('raw/Crimes-2018.csv')
    crimes = pd.concat([crime17, crime18])

    crimes['coords'] = list(zip(crimes.Longitude, crimes.Latitude))
    crimes['coords'] = crimes['coords'].apply(Point)
    crimes = gpd.GeoDataFrame(crimes, geometry = 'coords')
    print(crimes.head())

    return crimes



if __name__ == "__main__":
    
    census_data = get_census_data()
    crimes = load_crime_data()
    cook_blks = prep_census_shp()
    cook_blks = cook_blks.merge(census_data, on='bg')

