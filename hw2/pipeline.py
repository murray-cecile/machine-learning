#==============================================================================#
# MACHINE LEARNING FOR PUBLIC POLICY - PIPELINE
#
# Cecile Murray
#==============================================================================#

# dependencies
import numpy as np
import pandas as pd 
import seaborn as sns
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score as accuracy
import graphviz 


# globals
NUMERIC_TYPES = set(('int', 'float', 'int64', 'float64'))

#==============================================================================#
# 1. READ IN DATA
#==============================================================================#


# TO DO: implement additional options
def read_data(filename, file_type):
    '''Takes string name of csv or excel file (for now) and returns pandas data frame 
        If a data dict is provided it will rename vars'''
    
    if file_type == "csv":
        return pd.read_csv(filename)
    elif file_type == "excel": 
        return pd.read_excel(filename)
    else:
        print("filetype not supported")
        return

#==============================================================================#
# 2. EXPLORE DATA
#==============================================================================#

def get_desc_stats(df, *cols):
    ''' compute basic descriptive stats for any number of specified columns (as string)
        if none provided, computes only for numeric type columns'''

    if cols:
        return df[df[list(cols)]].describe()
    else:
        numeric_cols = df.columns.dtypes[]
        # TO DO: implement this using df.dtypes


 def find_outliers(df, var, lb, ub):
    ''' Checks whether all values of a variable fall within reasonable bounds '''

    too_small = df[var].loc[df[var] < lb]
    too_big = df[var].loc[df[var] > ub]

    print('# of values that are too small: ', len(too_small.index))
    print(too_small.head())
    print('# of values that are too large:', len(too_big.index))
    print(too_big.head())

    return 


def plot_distr(df, *cols):
    ''' Create histograms of numeric variables in dataframe; 
        optionally specify which variables to use '''

    if not cols:
        cols = df.columns
    
    if len(cols) == 1:
        cols = [cols[0]]

    # this part is still plotting on top of everything
    for c in cols:
        if str(df[c].dtype) in NUMERIC_TYPES:
            sns.distplot(df[c].loc[df[c].notnull()], kde = False)    
    
    return


def compute_pct_diff(df, var1, var2):
    
    pass


def tab(df, var1):

    return df.groupby(var1).count()


#==============================================================================#
# 3. PROCESS DATA
#==============================================================================#

def replace_missing(df, var, method = 'median'):
    ''' Takes data frame, column name, and optional replacement method;
        replaces missing values either with median (default) or with specified value;
        returns data frame '''

    if method == 'median': 
        df[var] = df[var].fillna(df[var].median())
    
    elif method == "mean":
        df[var] = df[var].fillna(df[var].mean())

    else:
        pass
    
    return df


#==============================================================================#
# 4. GENERATE FEATURES
#==============================================================================#

def discretize(df, var, breaks = [], num_breaks = False):
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


#==============================================================================#
# 5. BUILD CLASSIFIER
#==============================================================================#

def create_train_test_sets(df, target, features, size):
    ''' wrapper for scikitlearn's train_test_split 
        Returns: x train, x test, y train, y test'''

    y = df[target]
    x = df[features]

    return train_test_split(x, y, test_size=size)


def build_decision_tree(x_train, y_train, *kwargs):
    ''' wrapper to scikitlearn's build decision tree 
        Returns: DecisionTreeClassifier object'''

    return DecisionTreeClassifier(*kwargs).fit(x_train, y_train)



#==============================================================================#
# 6. EVALUATE
#==============================================================================#


def make_tree_histogram(dec_tree):
    '''CITE '''

    predicted_scores_test = dec_tree.predict_proba(x_test)[:,1]
    return plt.hist(predicted_scores_test)


def compute_accuracy(thresh, y_test):
    '''CITE'''

    calc_threshold = lambda x,y: 0 if x < y else 1 
    predicted_test = np.array([calc_threshold(score, thresh) for score in predicted_scores_test])
    test_acc = accuracy(predicted_test, y_test)

    return test_acc


def get_feature_wt(dec_tree, feature_list):
    ''' returns dict mapping feature names to weights'''

    return dict(zip(feature_list, list(dec_tree.feature_importances_)))


def test_tree_depths(depths = list(range(1, 6)), *kwargs):
    ''' Test different depths for the tree; default gives depths 1-5
        Returns: some nice way of comparing'''

    for d in depths:

        dec_tree = build_decision_tree