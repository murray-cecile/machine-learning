#==============================================================================#
# HW #3 - PREDICT FUNDING STATUS OF DONORSCHOOSE PROJECTS
# CAPP 30254
#
# Cecile Murray
#==============================================================================#

# basic dependencies
import yaml
import datetime
import numpy as np
import pandas as pd 
import seaborn as sns
import plotnine as p9
import matplotlib.pyplot as plt
from sklearn import preprocessing

# my own modules
import pipeline.utils as utils
import pipeline.exploration as exp
import pipeline.modeling as pipe


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
    df = utils.make_cat_dummy(df, categorical_list)

    # create major city dummy for biggest cities
    big_cities = ['Los Angeles', 'Chicago', 'Houston', 'Brooklyn', 'Bronx', 'New York']
    df['in_big_city'] = np.where(df['school_city'].isin(big_cities), 1, 0)
    

    # create dummies for female teacher and teacher with doctorate
    df['teacher_is_female'] = np.where(df['teacher_prefix'].isin(['Mrs.', 'Ms.']), True, False)
    df['teacher_is_dr'] = np.where(df['teacher_prefix'] == 'Dr.', True, False)

    # replace string true/false with boolean values
    df = utils.convert_to_boolean(df, ['school_charter',
                                        'school_magnet',
                                        'eligible_double_your_impact_match'], 't', 'f')

    return df


def normalizer_func(df):
    ''' Application-specific function to replace NA's in students reached and
        to normalize project price and students reached
    '''

    df['students_reached'] = df['students_reached'].fillna(0)
    df['total_price_norm'] = preprocessing.scale(df['total_price_including_optional_support'].astype('float64'))
    df['students_reached_norm'] = preprocessing.scale(df['students_reached'].astype('float64'))

    return df


def run_models(config):

    with open(config, 'r') as f:
        config = yaml.safe_load(f.read())

    # read in raw data
    raw_df = utils.read_data(config['raw_data_file'], 'csv')

    # convert date fields and generate target variable
    df = convert_dates(raw_df)

    # create training and testing sets: expanding window cross-validation
    df = pipe.create_sliding_window_sets(df,
                                        'date_posted',
                                        'not_funded',
                                        26)

    # clean the data (operations affect only each row in isolation)
    df = prepare_data(df)

    # loop over intervals: perform normalizations/interpolations/discretization, make dummies
    for i in pipe.get_date_intervals(df, 'interval'):
        df = utils.transform_vars_safely(df,
                                            normalizer_func,
                                            'date_posted',
                                            'projectid',
                                            i.left,
                                            i.right)

    # generate list of features after transformations are complete
    not_feature_cols = config['cols_to_exclude']
    features = list(set(df.columns).difference(not_feature_cols))

    # run all models across 7 thresholds
    results = pipe.test_over_time(df,
                                features,
                                'not_funded', 
                                'interval', 
                                'date_posted', 
                                lag_time=datetime.timedelta(days=60),
                                to_file = config['results_file'],
                                classifier_dict = config[config['MODEL_GRID']],
                                percentiles = config['PERCENTILES'])


def analyze_results(thresh = None):
    ''' Summarize results of model performance scores'''

    # read in model performance results
    results = pd.read_csv(config['results_file'])

    # segment to just a specific subset of results
    if thresh:
        results = results.loc[results['Threshold'] == thresh]

    # Create table showing precision, recall, and accuracy 
    metrics = ['classifier',
               'params',
               'Train/Test Split ID',
               'Precision',
               'Recall',
               'Accuracy',
               'AUC_ROC Score']
    results[metrics].groupby(['classifier', 'params', 'Train/Test Split ID']).mean().to_csv('output/Memo_Table.csv')

    return results, metrics


if __name__ == "__main__":

    run_models("config.yml")
    results, metrics = analyze_results(thresh = 0.05)
