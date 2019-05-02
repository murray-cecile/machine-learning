#==============================================================================#
# BUILD AND EVALUATE CLASSIFIERS
# CAPP 30254 - MACHINE LEARNING FOR PUBLIC POLICY
#
#   REFERENCES:
#   CAPP 30254 labs: https://github.com/dssg/MLforPublicPolicy/blob/master/labs/2019/
#
# Cecile Murray
#==============================================================================#

 # basic dependencies
import math
import datetime
import numpy as np
import pandas as pd 
import seaborn as sns
import plotnine as p9
import matplotlib.pyplot as plt

# my own library of useful functionsD
import utils

# classifiers
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier

# evaluation methods
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import precision_recall_curve 
from sklearn.utils.fixes import signature
import graphviz 

#==============================================================================#
# GLOBAL DEFAULTS
#==============================================================================#

THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

CLASSIFIERS = {
    'DecisionTree': {'max_depth': [1, 3, 5, 10, 15],
                     'criterion': ['gini', 'entropy']
                     },
    'KNN': {'k' = [3, 5, 10, 15, 25, 50]
            },
    'LogisticRegression': {'penalty': ['l1', 'l2'],
                            'C': [0.1, 1, 10, 100]
                            },
    'SVM': {'penalty': ['l1', 'l2'],
            'C' = [0.1, 1, 10, 100]
            },
    'BA': {'n_estimators': [10, 25, 100],
           'max_depth': [1, 3, 5, 10, 15]
            },
    'GB': {'n_estimators': [10, 25, 100],
            'max_depth': [1, 3, 5, 10, 15]
            },
    'RandomForest': {'n_estimators': [1, 10, 50, 100],
                     'max_depth': [1, 3, 5, 10, 15],
                     'criterion': ['gini', 'entropy']
                     }
}


#==============================================================================#
# BUILD TRAINING AND TEST SETS
#==============================================================================#

def create_train_test_sets(df, target, features, size = 0.25):
    ''' wrapper for scikitlearn's train_test_split 
        Returns: x train, x test, y train, y test'''

    y = df[target]
    x = df[features]

    return train_test_split(x, y, test_size=size)


def create_temporal_split_sets(df, date_col, feature_list, target, split_date):
    ''' Takes full dataframe, string name of date columns, list of features,
        string name of target variable, and date to split
        Returns: training and test data frames (before/after split date)
    '''

    df[date_col] = pd.to_datetime(df[date_col])

    x_train = df[feature_list].loc[df[date_col] < split_date]
    y_train = df[target].loc[df[date_col] < split_date]
    x_test = df[feature_list].loc[df[date_col] >= split_date]
    y_test = df[target].loc[df[date_col] >= split_date]

    return x_train, y_train, x_test, y_test


def convert_duration_to_interval(df, date_col, time_interval, time_unit = "weeks", end_with_max = True):
    ''' Takes dataframe, string name of date column, number of intervals
        Will eventually allow different time intervals
        Default option ends last interval with the latest date in the dataframe (so last chunk unequal)

        Returns: list of dates demarcating time intervals
    '''

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    intervals = [min_date]

    if time_unit == "weeks":

        interval_length = datetime.timedelta(weeks = time_interval)
        num_intervals = math.floor((max_date - min_date) / interval_length)

        i = 0
        next_date = min_date
        while i < num_intervals:
            next_date = next_date + interval_length
            intervals.append(next_date)
            i += 1

        # final interval will end with the final date u
        if end_with_max and intervals[-1] < max_date :
            intervals[-1] = max_date

        return intervals
    
    else:
        print("Time unit not yet supported; please convert to weeks")
        return


def create_sliding_window_sets(df, date_col, feature_list, target, time_interval, lag_time):
    ''' Takes full dataframe, string name of date column, list of features, 
        string name of target variable, number of intervals, and any lag time.
        Returns dataframe with bins corresponding to interval membership
    '''

    df[date_col] = pd.to_datetime(df[date_col])

    intervals = convert_duration_to_interval(df, date_col, time_interval)
    df['interval'] = pd.cut(df[date_col], intervals)
   
    # we don't want to include any observations too close to train/test date,
    # if we haven't yet observed their outcome
    df['interval'] = np.where(df[date_col] + lag_time > df['interval'].apply(lambda x: x.right), np.nan, df.interval)
    
    return df


def create_expanding_window_sets(df, date_col, feature_list, target, time_interval, lag_time):
    ''' Takes full dataframe, string name of date column, list of features, 
        string name of target variable, number of intervals, and any lag time.
        Returns dataframe with bins corresponding to interval membership
    '''

    df[date_col] = pd.to_datetime(df[date_col])

    breaks = convert_duration_to_interval(df, date_col, time_interval)
    intervals = [pd.Interval(breaks[0], b) for b in breaks[1:]]

    for i in range(0, len(intervals)):
        int_name = 'interval_' + str(i)
        df[int_name] = df[date_col].apply(lambda x: x in intervals[i])
        df[int_name] = np.where(df[date_col] + lag_time > intervals[i].right, np.nan, df[int_name])
       
    return df


#==============================================================================#
# BUILD CLASSIFIERS
#==============================================================================#

def build_classifier(classifier_type, x_train, y_train, **params):
    ''' Takes specified type of classifier using training set and optional keyword arguments
        Returns the trained classifier object
    '''

    if classifier_type == 'DecisionTree':
        return DecisionTreeClassifier(**params).fit(x_train, y_train)

    elif classifier_type == "LogisticRegression":
        return LogisticRegression(**params).fit(x_train, y_train)
    
    elif classifier_type == "KNN":
        return KNeighborsClassifier(**params).fit(x_train, y_train)
    
    elif classifier_type == "SVM":
        return LinearSVC(**params).fit(x_train, y_train)

    elif classifier_type == "BA":
        return BaggingClassifier(**params).fit(x_train, y_train)

    elif classifer == "GB":
        return GradientBoostingClassifier(**params).fit(x_train, y_train) 

    elif classifier_type == "RandomForest":
        return RandomForestClassifier(**params).fit(x_train, y_train)

    else:
        print("Classifier not supported")
        return


#==============================================================================#
# EVALUATE CLASSIFIERS
#==============================================================================#

def get_feature_wt(dec_tree, feature_list):
    ''' returns dict mapping feature names to weights
    '''

    return dict(zip(feature_list, list(dec_tree.feature_importances_)))


def compute_eval_stats(classifier, x_data, y_data, threshold):
    ''' Takes: classifier object, feature and target data, and
                prediction score threshold
        Returns: accuracy, precision, recall of predictions of classifier on x for y
    '''

    if type(classifier) == sklearn.svm.classes.LinearSVC:
        pred_scores = classifier.decision_function(x_data)[:,1]
    else:
        pred_scores = classifier.predict_proba(x_data)[:,1]
    
    calc_threshold = lambda x,y: 0 if x < y else 1 
    predicted_test = np.array( [calc_threshold(score, threshold) for score in pred_scores] )

    stats = [accuracy(y_data, predicted_test),
            precision(y_data, predicted_test),
            recall(y_data, predicted_test),
            f1(y_data, predicted_test),
            roc(y_data, predicted_test)]

    return stats


def draw_precision_recall_curve(classifier, x_data, y_data):
    ''' Takes: classifier object, feature and target data, 
        Returns: chart showing precision recall curve 

        Reference: code drawn from Scikit-learn documentation, https://bit.ly/2WaYP2I
    '''

    pred_scores = classifier.predict_proba(x_data)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_data, pred_scores)

    step_kwargs = ({'step': 'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')

    return

#==============================================================================#
# TEST DIFFERENT PARAMETERS
#==============================================================================#

def test_thresholds(classifier, x_data, y_data, threshold_list = []):
    ''' Takes classifier object, feature and target data, and list of score thresholds
        Returns: data frame summarizing performance for each threshold level
    '''

    results = []
    cols = ['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC Score']

    if not threshold_list:
        threshold_list = THRESHOLDS

    for t in threshold_list:

        stats = compute_eval_stats(classifier, x_data, y_data, t)
        results.append(t, [stats[0], stats[1], stats[2], stats[3], stats[4]])
    
    return pd.DataFrame(results, columns = cols)


def test_classifier_parameters(classifier_type, x_train, y_train, x_test, y_test, test_params, thresholds = []):
    ''' Test different parameters for a given classifier
        Returns: data frame summarizing model performance scores
    '''

    results = []

    for p in ParameterGrid(test_params):

        classifier = build_classifier(classifier_type, x_train, y_train, **p)
        test_performance = test_thresholds(classifier, x_test, y_test, thresholds)
        test_performance['params'] = str(p)

        results.append(test_performance)

    return pd.concat(results) 
 

def test_classifiers(x_train, y_train, x_test, y_test, classifier_dict = {}, thresholds = []):
    ''' Takes training and test data, and optionally classifier types and parameters
        and score thresholds (defaults set in globals)

        Returns: data frame summarizing performance of all combinations
    '''

    if not classifier_dict:
        classifier_dict = CLASSIFIERS
    classifier_list = classifier_dict.keys()

    results = []

    for c in classifier_list:

        test_params = classifier_dict[c] 
        performance = test_classifier_parameters(c, x_train, y_train, x_test, y_test, test_params, thresholds)
        performance['classifier'] = c

        results.append(performance)
    
    return pd.concat(results)


def test_over_time(df, features, target, intervals, classifier_dict = {}, thresholds = []):
    ''' Takes data, feature list, target variable, list of intervals, and optional
        classifier types/parameters and score thresholds

        Returns: data frame summarizing performance across all train/test windows
    '''

    results = []

    for i in intervals:
        pass

    return

#==============================================================================#
# VISUALIZATION TOOLS
#==============================================================================#

def plot_prediction_distribution(classifier, feature_set):
    '''Takes classifier object and associated feature set
        Returns plot of predicted probability distribution '''

    if type(classifier) == sklearn.svm.classes.LinearSVC:
        predicted_scores = classifier.decision_function(feature_set)[:,1]
    else:
        predicted_scores = classifier.predict_proba(feature_set)[:,1]
    
    return plt.hist(predicted_scores)


def make_tree_chart(dec_tree, feature_labels, target_names, out_file = ''):
    ''' Creates a visualization of a decision tree '''

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