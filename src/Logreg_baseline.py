'''
Fit baseline logisitic regression
'''
import glob
import os
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
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

def main(file):
    file = pd.read_csv('stock_w_sentiment_20.csv')
    file = file.drop(labels=['headline_HE','headline_LM','headline_QDAP','snippet_HE', 'snippet_LM','snippet_QDAP'], axis=1)

    list_sentiment = []
    for filename in glob.glob("end-pca/*.csv"):
        print(filename)
        sentiment = pd.read_csv(filename)
        sentiment = pd.DataFrame(sentiment[['date','headline_vector_pca','snippet_vector_pca']])
        sentiment = sentiment.drop_duplicates(subset=['date'], keep='first')
        sentiment["Ticker"] = filename.split("/")[1].split("_")[0]
        list_sentiment.append(sentiment) 
    list_sentiment = pd.concat(list_sentiment,axis=0)
    list_sentiment.to_csv("list_sentiment.csv", index=False)
    #read all sentiment files 
    signals = []
    signals_h = []
    signals_s = []
    #sentiment['headline_vector_pca'] = sentiment['headline_vector_pca'].str.split(',')
    #sentiment['snippet_vector_pca'] = sentiment['snippet_vector_pca'].str.split(',')
    #sentiment
    for row in list_sentiment['headline_vector_pca']:
        v = row.split(",")
        v = [float(x) for x in v]
        signals_h.append(v)
    for row in list_sentiment['snippet_vector_pca']:
        v = row.split(",")
        v = [float(x) for x in v]
        signals_s.append(v)
    for i in range(len(signals_h)):
        signal = signals_h[i] + signals_s[i]
        signals.append(signal)
    signals = pd.DataFrame(signals)
    signals = signals.reset_index(drop=True)
    signals
    list_sentiment = list_sentiment.reset_index(drop=True)

    list_sentiment = list_sentiment.join(signals)
    list_sentiment.rename(columns={'date':'Date'}, inplace=True)
    file.Date = pd.to_datetime(file.Date) 
    list_sentiment.Date = pd.to_datetime(list_sentiment.Date) 
    list_sentiment.to_csv("list_sentiment.csv", index=False)
    file = pd.merge(file, list_sentiment, on = ['Date', 'Ticker'], how = "left")
    #file = pd.merge(file, tech, on = ['Date','Ticker'], how = "left")
    file.to_csv("output.csv", index=False)

    saved_file = file
    
    m, n = file.shape
    daily = file['Adj Close'].shift(-1)/file['Adj Close'] - 1
    y = np.zeros(m)
    for i in range(m):
        if daily[i] >= 0:
            y[i] = 1 # up
        else:
            y[i] = 0 # down
    file['y-label'] = y
    file.to_csv('nyt.csv')
    #encode the ticker and sector into integer categories for our logistic regression vector, add y -> next day bucket
    file['Ticker'] = pd.Categorical(file['Ticker'])
    file['Sector'] = pd.Categorical(file['Sector'])
    categorical = ['Ticker','Sector']
    file = pd.get_dummies(file, columns=categorical, drop_first=True)
    #for libor, just use yesterday's if no value
    file['Libor'].fillna(method='pad', inplace=True)
    for i in range(0, 40):
        file[i].fillna(method='pad', inplace=True)
    #drop the rows with no value change (first 30 days)
    file = file.dropna(subset=['Thirty-day Change (%)'])
    file.head()
    file = file.drop(labels=['Unnamed: 0', 'Unnamed: 0.1','Unnamed: 0.1.1','High','Low', 'Open', 'Close', 'Adj Close','headline_vector_pca','snippet_vector_pca'], axis=1)
    file.head()
    file.fillna(method='pad', inplace=True)
    file.head()
    file.to_csv('test.csv')
    
    # Data preprocessing
    file.Date = pd.to_datetime(file.Date)
    file['y-label'] = pd.Categorical(file['y-label'])
    file_train = file[(file['Date'] <'2017-01-01')]
    file_test = file[(file['Date'] >='2017-01-01')]
    y_train = file_train['y-label']
    y_test = file_test['y-label']
    X_train = file_train.drop(labels=['Date', 'y-label'], axis=1)
    X_test = file_test.drop(labels=['Date', 'y-label'], axis=1)
    X_train['Volume'] = np.log(X_train['Volume']+0.5)
    X_test['Volume'] = np.log(X_test['Volume']+0.5)
   
    baseline_LR = LogisticRegression(C = 1e10, tol=0.000000001, max_iter=100000)
    baseline_LR.fit(X_train, y_train)
    # print('Coefficients: \n', baseline_LR.coef_)
    print("training accracy: %.4f" % baseline_LR.score(X_train, y_train))
    print("test accuracy: %.4f" % baseline_LR.score(X_test, y_test))
    
    y_fit_LR = baseline_LR.predict(X_test)

    print('Accuracy:', accuracy_score(y_test, y_fit_LR))
    print('ROC AUC Score:', roc_auc_score(y_test, y_fit_LR))
    print('F1 score:', f1_score(y_test, y_fit_LR))
    print()
    print('\n clasification report:\n', classification_report(y_test, y_fit_LR))
    print('\n confussion matrix:\n',confusion_matrix(y_test, y_fit_LR))
    
if __name__ == '__main__':
    main('goog_avg.csv') # Input Google or NYtimes news
