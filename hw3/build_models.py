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


def convert_dates(df):
    ''' Convert date fields to datetime and compute target variable:
        project fails to be funded within 60 days of being posted on DonorsChoose
    '''

    # convert date fields and compute whether project is funded within 60 days
    df['date_posted'] = pd.to_datetime(df.date_posted) 
    df['datefullyfunded'] = pd.to_datetime(df.datefullyfunded)
    df['not_funded'] = np.where(df.datefullyfunded - df.date_posted > datetime.timedelta(days=60), 1, 0)

    return df


def prepare_data(df):
    ''' Perform a range of application-specific data cleaning tasks '''

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

    return proj


def normalizer_func(df):
    ''' Application-specific function to replace NA's in students reached and
        to normalize project price and students reached
    '''

    df['students_reached'] = df['students_reached'].fillna(0)
    df['total_price_norm'] = preprocessing.scale(df['total_price_including_optional_support'].astype('float64'))
    df['students_reached_norm'] = preprocessing.scale(df['students_reached'].astype('float64'))

    return df


if __name__ == "__main__":

    # read data in    
    projraw = utils.read_data('projects_2012_2013', 'csv')

    # convert date fields and label target variable
    proj = convert_dates(projraw)

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
    proj = pipe.create_sliding_window_sets(projraw,
                                            'date_posted',
                                            features,
                                            'not_funded',
                                            26)

    # clean the data (operations affect only each row in isolation)
    proj = prepare_data(projraw)

    # loop over intervals and perform normalizations
    for i in pipe.get_date_intervals(proj, 'interval'):
        proj = utils.transform_vars_safely(proj,
                                            normalizer_func,
                                            'date_posted',
                                            'projectid',
                                            i.left,
                                            i.right)

    # proj.drop(columns = not_feature_cols)
    features = list(set(proj.columns).difference(not_feature_cols))

    # run all 165 models across 7 thresholds
    pipe.test_over_time(proj,
                        features,
                        'not_funded', 
                        'interval', 
                        'date_posted', 
                        lag_time=datetime.timedelta(days=60),
                        to_file = 'revised_test_run.csv',
                        classifier_dict = pipe.TEST_CLASSIFIERS)
