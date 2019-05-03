#==============================================================================#
# HW #3 - PREDICT FUNDING STATUS OF DONORSCHOOSE PROJECTS
# CAPP 30254
#
# Cecile Murray
#==============================================================================#

# basic dependencies
import datetime
import numpy as np
import pandas as pd 
import seaborn as sns
import plotnine as p9
import matplotlib.pyplot as plt
from sklearn import preprocessing

# my own library of useful functions
import utils
import exploration as exp
import pipeline as pipe


def prepare_data(df):

        # convert categorical variables to binary
    categorical_list = ['school_metro',
                        'school_state',
                        'primary_focus_subject',
                        'primary_focus_area',
                        'secondary_focus_subject',
                        'secondary_focus_area', 
                        'resource_type', 
                        'poverty_level',
                        'grade_level']
    proj = utils.make_cat_dummy(projraw, categorical_list)

    # create major city dummy for biggest cities
    big_cities = ['Los Angeles', 'Chicago', 'Houston', 'Brooklyn', 'Bronx', 'New York']
    proj['in_big_city'] = np.where(proj['school_city'].isin(big_cities), 1, 0)

    # create dummies for female teacher and teacher with doctorate
    proj['teacher_is_female'] = np.where(proj['teacher_prefix'].isin(['Mrs.', 'Ms.']), True, False)
    proj['teacher_is_dr'] = np.where(proj['teacher_prefix'] == 'Dr.', True, False)

    # replace string true/false with boolean values
    proj = utils.convert_to_boolean(proj, ['school_charter',
                                        'school_magnet',
                                        'eligible_double_your_impact_match'], 't', 'f')

    # replace NAs in students reached
    proj['students_reached'] = proj['students_reached'].fillna(0)

    # normalize project price and students reached
    proj['total_price_norm'] = preprocessing.scale(proj['total_price_including_optional_support'].astype('float64'))
    proj['students_reached_norm'] = preprocessing.scale(proj['students_reached'].astype('float64'))

    # convert date fields and compute whether project is funded within 60 days
    proj['date_posted'] = pd.to_datetime(proj.date_posted) 
    proj['datefullyfunded'] = pd.to_datetime(proj.datefullyfunded)
    proj['not_funded'] = np.where(proj.datefullyfunded - proj.date_posted > datetime.timedelta(days=60), 1, 0)

    return df


if __name__ == "__main__":

    # read data in    
    projraw = utils.read_data('projects_2012_2013', 'csv')

    # clean the data
    proj = prepare_data(projraw)

    # select features by set difference
    not_feature_cols = ['teacher_acctid',
                    'schoolid',
                    'school_ncesid',
                    'school_latitude', 
                    'school_longitude',
                    'school_district',
                    'school_county',
                    'school_city',
                    'projectid',
                    'teacher_prefix',
                    'date_posted',
                    'datefullyfunded',
                    'not_funded',
                    'interval',
                    'bins',
                    'students_reached',
                   'total_price_including_optional_support']
    features = list(set(proj.columns).difference(not_feature_cols))

    # create training and testing sets: expanding window cross-validation
    lag_time = datetime.timedelta(days=60)
    proj = pipe.create_sliding_window_sets(proj, 'date_posted', features, 'not_funded', 26)

    # run all 165 models across 7 thresholds
    pipe.test_over_time(proj, features, 'not_funded', 'interval', 'date_posted', lag_time=lag_time,
                    to_file = 'full_run.csv', classifier_dict = pipe.CLASSIFIERS)
