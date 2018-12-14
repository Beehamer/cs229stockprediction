'''
Fit logisitic regression and SVM on google news data
'''
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def calc_accuracy(pred, y):
    pred = np.matrix(pred).reshape(-1, 1)
    y = np.matrix(y).reshape(-1, 1)
    count = y.shape[0]
    correct = 0
    for i in range (count):
        if pred[i,0] == y[i, 0]:
            correct += 1
    return (correct * 1.0 / count)

def main():
    # Load data
    full = pd.read_csv('goog_avg.csv')
    full = full.drop(labels=['Unnamed: 0'], axis=1)
    full.head()
    full['Ticker'] = pd.Categorical(full['Ticker'])
    full['Sector'] = pd.Categorical(full['Sector'])
    #for libor, just use yesterday's if no value
    full['Libor'].fillna(method='pad', inplace=True)
    #for news, just use yesterday's if no value
    for i in range(0, 40):
        full[str(i)].fillna(method='pad', inplace=True)
    #drop the rows with no value change (first 30 days)
    full = full.dropna(subset=['Thirty-day Change (%)'])
    full = full.drop(labels=['High','Low', 'Open', 'Close', 'Adj Close'], axis=1)
    full.head()
    categorical = ['Ticker','Sector']
    file_one_hot_encoded = pd.get_dummies(full, columns=categorical, drop_first=True)
    in_set = file_one_hot_encoded
    in_set.Date = pd.to_datetime(in_set.Date)
    in_set['y-label'] = pd.Categorical(in_set['y-label'])
    in_train = in_set[(in_set['Date'] <'2017-01-01')]
    in_test = in_set[(in_set['Date'] >='2017-01-01')]
    y_train = in_train['y-label']
    y_test = in_test['y-label']
    X_train = in_train.drop(labels=['Date', 'y-label'], axis=1)
    X_test = in_test.drop(labels=['Date', 'y-label'], axis=1)
    X_train['Volume'] = np.log(X_train['Volume']+0.5)
    X_test['Volume'] = np.log(X_test['Volume']+0.5)

    # Fit and report Logistic Regression
    baseline_LR = LogisticRegression(C = 1e10, tol=0.000000001, max_iter=100000)
    baseline_LR.fit(X_train, y_train)
    # Training acc
    pred = baseline_LR.predict(X_train)
    accuracy = calc_accuracy(pred, y_train)
    print('The training accuracy of the SVM is ', accuracy)
    # Test acc
    y_hat = baseline_LR.predict(X_test)
    test_accuracy = calc_accuracy(y_hat, y_test)
    print('The test accuracy of the SVM is ', test_accuracy)
    # Conf matrix
    print('\n Training clasification report:\n', classification_report(y_train, pred))
    print('\n confusion matrix:\n',confusion_matrix(y_train, pred))
    print('\n Test clasification report:\n', classification_report(y_test, y_hat))
    print('\n confusion matrix:\n',confusion_matrix(y_test, y_hat))

    # Fit and report SVM
    # c = 0.5
    svm_rbf = svm.SVC(C=0.5, kernel='rbf',max_iter=50000)
    svm_rbf.fit(X_train, y_train)
    # Training acc
    pred = svm_rbf.predict(X_train)
    accuracy = calc_accuracy(pred, y_train)
    print('The training accuracy of the SVM is ', accuracy)
    # Test acc
    y_hat = svm_rbf.predict(X_test)
    test_accuracy = calc_accuracy(y_hat, y_test)
    print('The test accuracy of the SVM is ', test_accuracy)
    # Conf matrix
    print('\n Training clasification report:\n', classification_report(y_train, pred))
    print('\n confusion matrix:\n',confusion_matrix(y_train, pred))
    print('\n Test clasification report:\n', classification_report(y_test, y_hat))
    print('\n confusion matrix:\n',confusion_matrix(y_test, y_hat))
