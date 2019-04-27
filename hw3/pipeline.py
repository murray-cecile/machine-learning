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

# evaluation methods
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
import graphviz 


#==============================================================================#
# BUILD CLASSIFIERS
#==============================================================================#

def create_train_test_sets(df, target, features, size):
    ''' wrapper for scikitlearn's train_test_split 
        Returns: x train, x test, y train, y test'''

    y = df[target]
    x = df[features]

    return train_test_split(x, y, test_size=size)


def build_classifier(classifier_type, x_train, y_train, **params):
    ''' Takes specified type of classifier using x, y, and optional keyword arguments
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

#==============================================================================#
# EVALUATE CLASSIFIERS
#==============================================================================#


def get_feature_wt(dec_tree, feature_list):
    ''' returns dict mapping feature names to weights
    '''

    return dict(zip(feature_list, list(dec_tree.feature_importances_)))


def compute_eval_stats(classifier, x_data, y_data, threshold, labels = ''):
    ''' Takes: classifier object, feature and target data, and
                prediction probability threshold
        Returns: accuracy, precision, recall of predictions of classifier on x for y
    '''

    pred_scores = classifier.predict_proba(x_data)[:,1]
    calc_threshold = lambda x,y: 0 if x < y else 1 
    predicted_test = np.array( [calc_threshold(score, threshold) for score in pred_scores] )

    return accuracy(y_data, predicted_test), precision(y_data, predicted_test),
    recall(y_data, predicted_test), f1(y_data, predicted_test)



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