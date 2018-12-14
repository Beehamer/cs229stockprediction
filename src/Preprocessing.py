'''
Preprocess and merge the data
'''
import numpy as np
import pandas as pd
import datetime as datetime
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

np.random.seed(1)

# Tech indicator
# Commodity Channel Index
def CCI(data, ndays):
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),
                    name = 'CCI')
    data = data.join(CCI)
    return data
def RSI(data, ndays):
    delta = data['Close'].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    RolUp = pd.rolling_mean(dUp, ndays)
    RolDown = pd.rolling_mean(dDown, ndays).abs()
    RS = RolUp / RolDown
    RSI = pd.Series((100.0 - (100.0 / (1.0 + RS))),
                    name = 'RSI')
    data = data.join(RSI)
    return data
# Ease of Movement
def EVM(data, ndays):
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br
    EVM_MA = pd.Series(pd.rolling_mean(EVM, ndays), name = 'EVM')
    data = data.join(EVM_MA)
    return data
# Force Index
def ForceIndex(data, ndays):
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex')
    data = data.join(FI)
    return data

def main():
    # Clean GDP
    GDP = pd.read_csv('GDP.csv')
    GDP.Date = pd.to_datetime(GDP.Date)
    GDP = GDP.set_index('Date').resample('B').ffill().reset_index()
    lastGDP = pd.read_csv('lastGDP.csv')
    lastGDP.Date = pd.to_datetime(lastGDP.Date)
    lastGDP = lastGDP.set_index('Date').resample('B').ffill().reset_index()
    GDP = pd.merge(GDP, lastGDP, on = ['Date'], how = "left")

    # Clean CPI
    CPI = pd.read_csv('CPI.csv')
    CPI.Date = pd.to_datetime(CPI.Date)
    CPI = CPI.set_index('Date').resample('B').ffill().reset_index()
    lastCPI = pd.read_csv('lastCPI.csv')
    lastCPI.Date = pd.to_datetime(lastCPI.Date)
    lastCPI = lastCPI.set_index('Date').resample('B').ffill().reset_index()
    CPI = pd.merge(CPI, lastCPI, on = ['Date'], how = "left")

    # Clean Libor
    Libor = pd.read_csv('3M Libor.csv')
    Libor.Date = pd.to_datetime(Libor.Date)
    Libor = Libor.set_index('Date').resample('B').ffill().reset_index()
    Libor = Libor[pd.to_numeric(Libor['Libor'], errors='coerce').notnull()]

    # Merge
    joint = pd.merge(CPI, GDP, on = ['Date'], how = "left")
    joint = pd.merge(joint, Libor, on = ['Date'], how = "left")
    joint.to_csv('macro.csv')
    stock = pd.read_csv('stock_data.csv')
    stock.Date = pd.to_datetime(stock.Date)
    stock = pd.merge(stock, joint, on = ['Date'], how = "left")
    stock.to_csv('stock_w_macro.csv')

    # Add industry
    ind = pd.read_csv('companylist.csv')
    stock = pd.merge(stock, ind, on = ['Ticker'], how = "left")
    stock.to_csv('stock_w_macro.csv')

    # Merge tech indicator
    top_20 = ['FB','AAPL','GOOG','INTU','ALGN','AMZN','AAL','MSFT','AMAT','FOX',
          'SBUX','WDC','NFLX','COST','ADBE','EBAY','WDAY','LRCX','BIDU','PYPL']
    n1 = 20
    n2 = 14
    n3 = 14
    n4 = 1
    to_merge = pd.DataFrame({'Date' : [],'CCI': [],'RSI': [],'EVM': [],'Ticker' : [],'ForceIndex':[]})
    for ticker in top_20:
        print('Processing ',ticker)
        data = pdr.get_data_yahoo(ticker, start="2013-01-01", end="2017-12-31")
        data = pd.DataFrame(data)
        data.reset_index(inplace=True)
        Ticker_CCI = CCI(data, n1)
        Ticker_RSI = RSI(Ticker_CCI, n2)
        Ticker_EVM = EVM(Ticker_RSI, n3)
        Ticker_ForceIndex = ForceIndex(Ticker_EVM,n4)
        out = Ticker_ForceIndex[['Date','CCI','RSI','EVM','ForceIndex']]
        out['Ticker']=ticker
        to_merge = pd.concat([to_merge,out])
    to_merge.to_csv('techindicators_20.csv', encoding='utf-8', index=True)


