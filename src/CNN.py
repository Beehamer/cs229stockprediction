import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from sklearn.metrics import confusion_matrix


def main(filename) :
    full = pd.read_csv(filename)
    full = full.drop(labels=['Unnamed: 0'], axis=1)
    top_20 = ['FB','AAPL','GOOG','INTU','ALGN','AMZN','AAL','MSFT','AMAT','FOX',
              'SBUX','WDC','NFLX','COST','ADBE','EBAY','WDAY','LRCX','BIDU','PYPL']
    for elem in top_20:
        tmp = full.loc[full['Ticker']==elem]
        savename = 'ForRNN/' + elem +'.csv'
        tmp.to_csv(savename)
        print(savename)
        
    #filename = 'ForRNN/FB.csv'
train_confusion = np.zeros((2, 2))
test_confusion = np.zeros((2, 2))
import glob
for filename in glob.glob('ForRNN/*'): 
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
    m1, n1 = X_train.shape
    x_train = np.array(X_train).reshape(m1, n1, 1)
    m2, n2 = X_test.shape
    x_test= np.array(X_test).reshape(m2, n2, 1)
    batch_size = 128
    epochs = 80
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=x_train.shape[1:3]))
    
    #model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    
    #model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    """
    #model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=x_train.shape[1:3]))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    """
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    print(x_train.shape)
    print(filename)
    if x_train.shape[0] == 0:
        continue
    
    # Train model
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
    
    # Training report
    train_eval = model.evaluate(x_train, y_train, verbose=0)
    print('Training loss:', train_eval[0])
    print('Training accuracy:', train_eval[1])
    pred = model.predict(x_train)
    pred = (pred > 0.5)
    train_conf = confusion_matrix(y_train, pred)
    train_confusion += train_conf
    
    # Test report
    test_eval = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:',test_eval[1])
    y_hat = model.predict(x_test)
    y_hat = (y_hat > 0.5)
    test_conf = confusion_matrix(y_test, y_hat)
    test_confusion += test_conf
    print(test_conf)
    
    
if __name__ == '__main__':
    main('goog_avg.csv') # Input Google or NYtimes news
    