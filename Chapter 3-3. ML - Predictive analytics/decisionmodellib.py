# decisionmodellib.py

## Decision Tree Model Library

# Load libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


# Function to load dataset
def importdata(filename):
    balance_data = pd.read_csv(
        filename,
        sep=',',
        header=None
    )
    print("Dataset length: ", len(balance_data))
    print("Dataset shape: ", balance_data.shape)
    print("Dataset (first 5 rows): ", balance_data.head())
    return balance_data


# print(importdata(filename))

# Function to split predictors and target variable
def splitdataset(balance_data, X, Y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X, Y, X_train, X_test, y_train, y_test


# Function to create and train Decision Tree classifier
def train_using(criterion, random_state, md, msl, X_train, X_test, y_train):
    clf = DecisionTreeClassifier(
        criterion=criterion,  # Type of classifier (algorithm)
        random_state=random_state,  # Random seed value
        max_depth=md,  # Tree depth
        min_samples_leaf=msl  # Minimum number of samples in a leaf node
    )
    clf.fit(X_train, y_train)
    return clf


# Function for predictions
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values: ", y_pred)
    return y_pred


# Function to print confusion matrix, accuracy, and report
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
    print("Report: ", classification_report(y_test, y_pred))


# Function to visualize the Decision Tree
def plot_decision_tree(figsize, clf_object, feature_names, class_names):
    plt.figure(figsize=figsize)
    plot_tree(
        clf_object,
        filled=True,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True
    )
    plt.show()


# Execution
filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
df = importdata(filename)
X = df.values[:, 1:5]  # Predictors
Y = df.values[:, 0]  # Target variable
X, Y, X_train, X_test, y_train, y_test \
    = splitdataset(df, X, Y, 0.3, 1234)
clf = train_using('gini', 1234, 3, 5, X_train, X_test, y_train)
plot_decision_tree(
    (15, 10),
    clf,
    ['X1', 'X2', 'X3', 'X4'],
    ['L', 'B', 'R']
)
