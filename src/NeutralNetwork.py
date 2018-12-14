'''
Fit NN on NYtimes and google news data
'''
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import copy
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

def loss_func(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num = y_true.shape[0]
    corr = 0
    for i in range(num):
        if y_true[i] == y_pred[i]:
            corr += 1
    return corr * 1.0 / num

def get_conf_mat(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))

def evaluate(clf, X_train, y_train, X_test, y_test, verbose=True):
    y_hat_train = clf.predict(X_train)
    y_hat_test = clf.predict(X_test)
    training_accu = loss_func(y_hat_train, y_train)
    testing_accu = loss_func(y_hat_test, y_test)
    if verbose:
        print ('The training accuracy is %f' % training_accu)
        print ('The testing accuracy is %f' % testing_accu)
        get_conf_mat(y_test, y_hat_test)
        print ('----------')
    return (training_accu, testing_accu)

def main(filename):
    full = pd.read_csv(filename)
    full = full.drop(labels=['Unnamed: 0'], axis=1)
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
    categorical = ['Ticker','Sector']
    file_one_hot_encoded = pd.get_dummies(full, columns=categorical, drop_first=True)
    file_one_hot_encoded.Date = pd.to_datetime(file_one_hot_encoded.Date)
    file_one_hot_encoded['y-label'] = pd.Categorical(file_one_hot_encoded['y-label'])
    file_train = file_one_hot_encoded[(file_one_hot_encoded['Date'] <'2017-01-01')]
    file_test = file_one_hot_encoded[(file_one_hot_encoded['Date'] >='2017-01-01')]
    y_train = file_train['y-label']
    y_test = file_test['y-label']
    X_train = file_train.drop(labels=['Date', 'y-label'], axis=1)
    X_test = file_test.drop(labels=['Date', 'y-label'], axis=1)
    X_train['Volume'] = np.log(X_train['Volume']+0.5)
    X_test['Volume'] = np.log(X_test['Volume']+0.5)

    learning_rates = [0.0001,0.001, 0.01, 0.05, 0.1]
    ## the optimal hidden layer size is between the input size and the output size
    hiddens = [(50, 50, 10), (50, 2), (20, 20),(10, 10),(10, 10, 10)]

    best_model = None
    best_acc = 0
    best_params = None

    models = []
    for lrts in learning_rates :
        for hidden_layers in hiddens:
            clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layers, random_state=1, learning_rate_init = lrts, max_iter=1000)
            print ('finishing fitting...')
            clf.fit(X_train, y_train)
            (train_acc, test_acc) = evaluate(clf, X_train, y_train , X_test, y_test, True)
            models.append(copy.deepcopy(clf))
            if test_acc >= best_acc:
                best_model = copy.deepcopy(clf)
                best_params = (lrts, copy.deepcopy(hidden_layers))
                best_acc = test_acc
    evaluate(best_model, X_train, y_train, X_test, y_test, True)
    print (best_params)

if __name__ == '__main__':
    main('goog_avg.csv') # Input Google or NYtimes news
