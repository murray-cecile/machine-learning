#==============================================================================#
# MACHINE LEARNING FOR PUBLIC POLICY - PIPELINE
#
# Cecile Murray
#==============================================================================#

import numpy as np
import pandas as pd 


# TO DO: implement additional options
def read_data(filename, file_type):
    '''Takes string name of csv or excel file (for now) and returns pandas data frame 
        If a data dict is provided it will rename vars'''
    
    if file_type == "csv":
        return pd.read_csv(filename)
    else: 
        return pd.read_excel(filename)


def get_desc_stats(df, *cols):
    ''' compute basic descriptive stats for any number of specified columns (as string)
        if none provided, computes only for numeric type columns'''

    if cols:
        return df[df[*cols]].describe()
    else:
        pass
        # TO DO: implement this using df.dtypes

 
def plot_distr(df, *cols):

    pass


def compute_pct_diff(df, var1, var2):
    
    pass


def tab(df, var1):

    return df.groupby(var1).count()


def replace_missing(df, var, method = 'median'):
    ''' Takes data frame, column name, and optional replacement method;
        replaces missing values either with median (default) or with specified value;
        returns data frame '''

    if method == 'median:' 
        df[var] = df[var].fillna(df[var].median())
    
    elif method == "mean":
        df[var] = df[var].fillna(df[var].mean())

    else:
        pass
    
    return df


def discretize(df, var, breaks, num_breaks = False):
    ''' Convert continuous variable to discrete/categorical '''

    # TO DO: NaN for values outside the provided breaks, index label argument

    new_varname = var + '_disc'

    if num_breaks:
        df[new_varname] = pd.cut(df[var], num_breaks)
    
    else:
        df[new_varname] = pd.cut(df[var], np.array(breaks))

    return df


def make_dummy(df, var, true_vals, new_varname = ''):
    ''' Takes: a data frame, the name of a categorical variable, the values to
        be assigned True, optional new variable name
        Returns: data frame with dummy variable added '''

    if not new_varname:
        new_varname = var + '_dummy'
    
    df[new_varname] = df[var] in true_vals

    return df
