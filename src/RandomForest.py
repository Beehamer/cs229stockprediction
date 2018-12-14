#import everything
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def main(filename):
    filename = 'stock_w_sentiment.csv'
    file = pd.read_csv(filename)
    print(file.shape)
    
    # Data preprocessing
    #encode the ticker and sector into one-hot integer categories for our logistic regression vector
    file['headline_HE'].fillna("no headline")
    file['headline_LM'].fillna("no headline")
    file['headline_QDAP'].fillna("no headline")
    file['snippet_HE'].fillna("no snippet")
    file['snippet_LM'].fillna("no snippet")
    file['snippet_QDAP'].fillna("no snippet")
    file.Ticker = pd.Categorical(file.Ticker)
    file['ticker_code'] = file.Ticker.cat.codes
    file.Sector = pd.Categorical(file.Sector)
    file['sector_code'] = file.Sector.cat.codes
    file.headline_HE = pd.Categorical(file.headline_HE)
    file['headline_HE'] = file.headline_HE.cat.codes
    file.headline_LM = pd.Categorical(file.headline_LM)
    file['headline_LM'] = file.headline_LM.cat.codes
    file.headline_QDAP = pd.Categorical(file.headline_QDAP)
    file['headline_QDAP'] = file.headline_QDAP.cat.codes
    file.snippet_HE = pd.Categorical(file.snippet_HE)
    file['snippet_HE'] = file.snippet_HE.cat.codes
    file.snippet_LM = pd.Categorical(file.snippet_LM)
    file['snippet_LM'] = file.snippet_LM.cat.codes
    file.snippet_QDAP = pd.Categorical(file.snippet_QDAP)
    file['snippet_QDAP'] = file.snippet_QDAP.cat.codes
    file = file.drop_duplicates(subset=['Date', 'Ticker'])
    #file = pd.get_dummies(file,columns=['Ticker', 'Sector', 'headline_HE', 'headline_LM', 'headline_QDAP', 'snippet_HE', 'snippet_LM', 'snippet_QDAP'])
    #add in y-label
    y = np.zeros(file.shape[0])
    print(file['Adj Close'].shape)
    daily = file['Adj Close'].shift(-1)/file['Adj Close'] - 1
    for i in range(file.shape[0]):
        if daily.iloc[i] >= 0.01:
            y[i] = 3
        elif daily.iloc[i] >= 0.00:
            y[i] = 2
        elif daily.iloc[i] >= -0.01:
            y[i] = 1
        else:
            y[i] = 0
    file['y-label'] = y
    #for libor, just use yesterday's if no value
    file['Libor'].fillna(method='pad', inplace=True)
    #drop the rows with no value change (first 30 days)
    file = file.dropna()
    file.head()
    
    file.to_csv('stock_data_final_sent.csv', encoding='utf-8', index=True)
    #split for train and test set, everything before 2017 is train
    file_train = file.loc[file['Date'].apply(lambda x: x.split('-')[0])!= "2017"]
    file_test = file.loc[file['Date'].apply(lambda x: x.split('-')[0]) == "2017"]
    
    train_x = file_train.drop(labels=['Date', 'y-label', 'Unnamed: 0', 'Unnamed: 0.1', 'Ticker', 'Sector'], axis=1).values.astype(np.float32)
    train_y = file_train['y-label'].values.astype(np.float32)
    test_x = file_test.drop(labels=['Date', 'y-label', 'Unnamed: 0', 'Unnamed: 0.1', 'Ticker', 'Sector'], axis=1).values.astype(np.float32)
    test_y = file_test['y-label'].values.astype(np.float32)
    
    rf = RandomForestClassifier(n_estimators=250, oob_score=True, random_state=123456)
    rf.fit(train_x, train_y)

    predicted = rf.predict(test_x)
    accuracy = accuracy_score(test_y, predicted)
    
if __name__ == '__main__':
    main('stock_w_sentiment.csv') # Input Google or NYtimes news

