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
import numpy as np
import pandas as pd 
import seaborn as sns
import plotnine as p9
import matplotlib.pyplot as plt

# my own library of useful functions
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
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import precision_recall_curve 
from sklearn.utils.fixes import signature
import graphviz 


#==============================================================================#
# BUILD CLASSIFIERS
#==============================================================================#

def create_train_test_sets(df, target, features, size) = 0.25:
    ''' wrapper for scikitlearn's train_test_split 
        Returns: x train, x test, y train, y test'''

    y = df[target]
    x = df[features]

    return train_test_split(x, y, test_size=size)


def create_temporal_validation(df, target, features, interval, size = 0.25):
    ''' Create temporal validation function in your pipeline that can create training and test sets over time. You can choose the length of these splits based on analyzing the data. For example, the test sets could be six months long and the training sets could be all the data before each test set.
    '''
    pass


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

    else:
        print("Classifier not supported.")
        return 


def build_bagged_classifier(classifier_type, classifier_params, **bag_params):
    ''' Takes specific type of classifier, any parameters, and optional keyword arguments
        Returns the trained bagged classifier object 
    '''

    pass
    # if classifier_type == 'DecisionTree':
    #     classifier = DecisionTreeClassifier(**params)

    # elif classifier_type == "LogisticRegression":
    #     classifier = LogisticRegression(**params)
    
    # elif classifier_type == "KNN":
    #     classifier = KNeighborsClassifier(**params)
    
    # elif classifier_type == "SVM":
    #     classifier = LinearSVC(**params)

    # else:
    #     print("Classifier not supported.")
    #     return 

    # return BaggingClassifier(classifier, **bag_params)


def grow_random_forest(x_data, y_data, **params):
    ''' Takes feature and target data and optional parameters
        Returns: random forest classifier
    '''

    return RandomForestClassifier(**params).fit(x_data, y_data)

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

def test_thresholds(classifier, x_data, y_data, threshold_list):
    ''' Takes classifier object, feature and target data, and list of score thresholds
        Returns: data frame summarizing performance for each threshold level
    '''

    results = []
    cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC Score']

    for t in threshold_list:

        stats = compute_eval_stats(classifier, x_data, y_data, t)
        results.append([stats[0], stats[1], stats[2], stats[3], stats[4]])
    
    return pd.DataFrame(results, columns = cols)


def test_classifier_parameters(classifier_type, x_train, y_train, x_test, y_test, threshold, test_params, **fixed_params):
    ''' Test different parameters for a given classifier
        Returns: data frame summarizing model performance scores
    '''

    results = []

    for p in test_params:

        classifier = build_classifier(classifier, x_train, y_train, fixed_params)

        train_stats = compute_eval_stats(classifier, x_train, y_train, threshold)
        test_stats = compute_eval_stats(classifier, x_test, y_test, threshold)

        results.append([c, d, train_stats[0],
                                test_stats[0],
                                test_stats[1],
                                test_stats[2],
                                test_stats[3]])
    
    return pd.DataFrame(results, columns = ['Split Criterion', 'Depth',
                                             'Train Accuracy',
                                             'Test Accuracy', 
                                             'Test Precision',
                                             'Test Recall',
                                             'Test F1'])

#==============================================================================#
# VISUALIZATION TOOLS
#==============================================================================#

def plot_prediction_distribution(classifier, feature_set):
    '''Takes decision tree classifier object and associated feature set
        Returns plot of predicted probability distribution '''

    predicted_scores = classifier.predict_proba(feature_set)[:,1]
    return plt.hist(predicted_scores)


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