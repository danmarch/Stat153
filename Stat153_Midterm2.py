"""Code for the midterm project in Statistics 153 - Time Series at UC Berkeley,
Spring 2017.
@author: Daniel March
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sys

df = pd.read_csv('q1_train.csv')
df.Date = pd.to_datetime(df.Date)
df.set_index("Date", inplace=True)
plt.plot(df)
plt.title("Original Data")
plt.show()

def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

### Differenced Signal ###
df_diff = df - df.shift()
df_diff.dropna(inplace=True)
df_diff_2 = df - df.shift(52)
df_diff_2.dropna(inplace=True)
df_diff_3 = df_diff - df_diff.shift(52)
df_diff_3.dropna(inplace=True)

def plot_cf(ts,field):
    lag_acf = acf(field, nlags=20)
    lag_pacf = pacf(field, nlags=20)
    #Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(df_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(df_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.show()

plot_cf(df_diff_3,df_diff_3.activity)

# Find the (p,q) pair that minimizes the Akaike Information Criterion.
# The possible range will be [0,10] for both p and q (this is manually set).
# min_val = sys.maxsize
# p,q = None,None
# for i in range(6):
#     for j in range(6):
#         try:
#             new_aic = ARIMA(df_diff,order=(i,1,j)).fit(disp=-1).aic
#         except ValueError as e:
#             continue
#         if new_aic < min_val:
#             p,q = i,j



# p,q = 0,2
#
# arma_fit = SARIMAX(df,order=(p,1,q),seasonal_order=(1,1,1,52)).fit(disp=-1)
# prediction = arma_fit.predict(start=525,end=525+103,dynamic=True)
# plt.plot(df)
# plt.plot(prediction)
# plt.show()
