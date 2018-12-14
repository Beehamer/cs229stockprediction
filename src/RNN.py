import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
from keras.layers import Embedding
from sklearn.metrics import confusion_matrix
batch_size = 128
epochs = 20

def evaluate(x_train, y_train, x_test, y_test):
    # Training report
    train_eval = model.evaluate(x_train, y_train, verbose=0)
    print('Training loss:', train_eval[0])
    print('Training accuracy:', train_eval[1])
    pred = model.predict(x_train)
    pred = (pred > 0.5)
    train_conf = confusion_matrix(y_train, pred)
    print(train_conf)
    # Test report
    test_eval = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    y_hat = model.predict(x_test)
    y_hat = (y_hat > 0.5)
    test_conf = confusion_matrix(y_test, y_hat)
    print(test_conf)

def main(filename):
    full = pd.read_csv(filename)
    full = full.drop(labels=['Unnamed: 0.1','Unnamed: 0.1.1','Unnamed: 0.1.1.1', 'headline_vector_pca','snippet_vector_pca'], axis=1)
    top_20 = ['FB','AAPL','GOOG','INTU','ALGN','AMZN','AAL','MSFT','AMAT','FOX',
              'SBUX','WDC','NFLX','COST','ADBE','EBAY','WDAY','LRCX','BIDU','PYPL']
    for elem in top_20:
        tmp = full.loc[full['Ticker']==elem]
        savename = 'ForRNN/' + elem +'.csv'
        tmp.to_csv(savename)
        print(savename)
    
    # load the data
    ticker = pd.read_csv(filename)
    ticker = ticker.drop(labels=['Unnamed: 0','Ticker','Sector'], axis=1)
    
    #for libor, just use yesterday's if no value
    ticker['Libor'].fillna(method='pad', inplace=True)
    
    #for news, just use yesterday's if no value
    for i in range(0, 40):
        ticker[str(i)].fillna(method='pad', inplace=True)
    
    #drop the rows with no value change (first 30 days)
    ticker = ticker.dropna(subset=['Thirty-day Change (%)'])
    ticker = ticker.drop(labels=['High','Low', 'Open', 'Close', 'Adj Close'], axis=1)
    
    # Separating the data to train/test
    ticker.Date = pd.to_datetime(ticker.Date)
    ticker['y-label'] = pd.Categorical(ticker['y-label'])
    file_train = ticker[(ticker['Date'] <'2017-01-01')]
    file_test = ticker[(ticker['Date'] >='2017-01-01')]
    y_train = file_train['y-label']
    y_test = file_test['y-label']
    X_train = file_train.drop(labels=['Date', 'y-label'], axis=1)
    X_test = file_test.drop(labels=['Date', 'y-label'], axis=1)
    X_train['Volume'] = np.log(X_train['Volume']+0.5)
    X_test['Volume'] = np.log(X_test['Volume']+0.5)
    
    # reshape the data with an input channel to feed into RNN
    m1, n1 = X_train.shape
    x_train = np.array(X_train).reshape(m1, n1, 1)
    # x_train = np.array(X_train)
    m2, n2 = X_test.shape
    x_test= np.array(X_test).reshape(m2, n2, 1)
    # x_test = np.array(X_test)
    
    # construct the model
    model = Sequential()
    model.add(LSTM(128, dropout=0.8, recurrent_dropout=0.8,  kernel_regularizer='l2', activity_regularizer='l2', input_shape=x_train.shape[1:3]))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
    evaluate(x_train y_train, x_test, y_test)

    
    
if __name__ == '__main__':
    main('goog_avg.csv') # Input Google or NYtimes news
    
    

