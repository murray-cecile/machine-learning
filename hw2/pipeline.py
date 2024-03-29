#==============================================================================#
# MACHINE LEARNING FOR PUBLIC POLICY - PIPELINE
#
# Cecile Murray
#==============================================================================#

# dependencies
import numpy as np
import pandas as pd 
import seaborn as sns
import plotnine as p9
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
import graphviz 


#==============================================================================#
# 1. READ IN DATA
#==============================================================================#


# TO DO: implement additional options
def read_data(filename, file_type):
    '''Takes string name of csv or excel file (for now) and returns pandas data frame 
        If a data dict is provided it will rename vars'''
    
    if file_type == "csv":
        return pd.read_csv('data/' + filename + '.csv')
    elif file_type == "excel": 
        return pd.read_excel('data/' + filename + '.xlsx')
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
        return df.select_dtypes(include = np.number).describe()


def find_outliers(df, lb, ub, var):
    ''' Checks whether all values of variable(s) fall within reasonable bounds '''

    too_small = df[var].loc[df[var] < lb]
    too_big = df[var].loc[df[var] > ub]

    print('# of values smaller than lower bound: ', len(too_small.index))
    print(too_small.head().sort_values())
    print('# of values larger than upper bound:', len(too_big.index))
    print(too_big.head().sort_values(ascending = False))
    print('\n')

    return 


def plot_correlations(df, cols):
    '''Takes a data frame, a list of columns, and a pair of names for plot axes
        Returns a plot of pairwise correlations between all variables
    '''

    axisnames = ["Variable 1", "Variable 2", "Correlation"]

    corr_df = pd.DataFrame(df[cols].corr().stack()).reset_index()
    dict(zip(list(corr_df.columns), axisnames))
    corr_df.rename(columns = dict(zip(list(corr_df.columns), axisnames)), inplace = True)

    return (p9.ggplot(corr_df, p9.aes(axisnames[0], axisnames[1], fill=axisnames[2]))
        + p9.geom_tile(p9.aes(width=.95, height=.95)) 
        + p9.theme(
        axis_text_x = p9.element_text(rotation = 90))
        )


def plot_distr(df, *cols):
    ''' Create histograms of numeric variables in dataframe; 
        optionally specify which variables to use '''

    if not cols:
        cols = df.columns
    
    if len(cols) == 1:
        cols = [cols[0]]

    # this part is still plotting on top of everything
    for c in cols:
        if df[c].dtype == np.number:
            sns.distplot(df[c].loc[df[c].notnull()], kde = False)    
    
    return


def plot_cond_dist(df, y, *x):
    ''' Plot conditional distributiofn of x on categorical or binary y '''

    for v in x:
        sns.FacetGrid(df, col=y).map(plt.hist, v)



def tab(df, y, *x):
    ''' Compute summary statistics about y conditioned on categorical variable(s) x '''

    if len(x) == 0:
        return False
    
    else:
        return df.groupby(list(x))[y].describe()


#==============================================================================#
# 3. PROCESS DATA
#==============================================================================#

# some pre-processing to convert numeric NA's to the median of the column
def replace_missing(df, *args, method = 'mean'):
    ''' Takes data frame, replacement method, and arbitrary column names;
        replaces missing values using specified method;
        returns data frame '''

    if method == 'median': 
        for c in list(args):
            df[c] = df[c].fillna(df[c].median())
    
    elif method == "mean":
        for c in list(args):
            df[c] = df[c].fillna(df[c].mean())

    else:
        pass
    
    return df

#==============================================================================#
# 4. GENERATE FEATURES
#==============================================================================#

def bin_continuous(df, var, new_varname, breaks, labels = False):
    ''' Convert continuous variable to discrete/categorical '''


    # handle case where upper bound less than max of variable
    if breaks[-1] < df[var].max():
        breaks.append(df[var].max())
        
    # handle case where lower bound greater than min of variable
    if breaks[0] > df[var].min():
        breaks.insert(0, df[var].min())

    if labels:
        df[new_varname] = pd.cut(df[var], np.array(breaks), labels = labels)
    else:
        df[new_varname] = pd.cut(df[var], np.array(breaks))
    
    return df


def make_cat_dummy(df, cols):
    '''Convert categorical variable or list of categorical variables into binary '''

    return pd.get_dummies(df, dummy_na=True, columns=cols)



#==============================================================================#
# 5. BUILD CLASSIFIER
#==============================================================================#

def create_train_test_sets(df, target, features, size):
    ''' wrapper for scikitlearn's train_test_split 
        Returns: x train, x test, y train, y test'''

    y = df[target]
    x = df[features]

    return train_test_split(x, y, test_size=size)


#==============================================================================#
# 6. EVALUATE
#==============================================================================#

# This section draws heavily on code developed in ML Lab #2:  
# https://github.com/dssg/MLforPublicPolicy/blob/master/labs/2019/lab2_sklearn_dt_knn.ipynb

def plot_prediction_distribution(dec_tree, x_test):
    '''Takes decision tree classifier object and test feature set
        Returns plot of predicted probability distribution '''

    predicted_scores_test = dec_tree.predict_proba(x_test)[:,1]
    return plt.hist(predicted_scores_test)


def get_feature_wt(dec_tree, feature_list):
    ''' returns dict mapping feature names to weights
    '''

    return dict(zip(feature_list, list(dec_tree.feature_importances_)))


def compute_accuracy(dec_tree, x_data, y_data, threshold):
    ''' Takes: decision tree classifier object, feature and target data, and
                prediction probability threshold
        Returns: accuracy of predictions of tree on x for y
    '''

    pred_scores = dec_tree.predict_proba(x_data)[:,1]
    calc_threshold = lambda x,y: 0 if x < y else 1 
    predicted_test = np.array( [calc_threshold(score, threshold) for score in pred_scores] )
    return accuracy(predicted_test, y_data)


def test_tree_accuracy(x_train, y_train, x_test, y_test, threshold, depths = list(range(0,15, 3)), criterion = ["gini"]):
    ''' Test different parameters for the tree
        Returns: data frame summarizing train and test accuracy
    '''

    results = []

    for c in criterion:
        for d in depths:

            dec_tree = DecisionTreeClassifier(max_depth=d, criterion=c).fit(x_train, y_train)

            train_acc = compute_accuracy(dec_tree, x_train, y_train, threshold)
            test_acc = compute_accuracy(dec_tree, x_test, y_test, threshold)

            results.append([c, d, train_acc, test_acc])
    
    return pd.DataFrame(results, columns = ['Split Criterion', 'Depth', 'Train Accuracy', 'Test Accuracy'])


def make_tree_chart(dec_tree, feature_labels, target_names, out_file = ''):
    ''' Creates a visualization of the tree '''

    if not out_file:
        out_file = "tree.dot"

    viz = tree.export_graphviz(dec_tree,
                            feature_names=feature_labels,
                            out_file= 'figs/' + out_file,
                            class_names=target_names,
                            rounded=True,
                            filled=True)

    with open('figs/' + out_file) as f:
        dot_graph = f.read()
        graph = graphviz.Source(dot_graph)
        
    return graph