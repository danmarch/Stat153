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
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sys

def main(dataset):

    def plot_cf(ts,field):
        """NOTE: I did NOT write this function. It was taken from:
        http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
        """
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

    if dataset == 1:
        df = pd.read_csv('q1_train.csv')
        df.Date = pd.to_datetime(df.Date)
        df.set_index("Date", inplace=True)

        ### Differenced Signal ###
        df_diff = df - df.shift()
        df_diff.dropna(inplace=True)
        df_diff_2 = df - df.shift(52)
        df_diff_2.dropna(inplace=True)
        df_diff_3 = df_diff - df_diff.shift(52)
        df_diff_3.dropna(inplace=True)

        p,q = 0,2
        arma_fit = SARIMAX(df,order=(p,1,q),seasonal_order=(1,1,1,52)).fit(disp=-1)
        prediction = arma_fit.predict(start=525,end=525+103,dynamic=True)

        ### Plots the original data with the prediction ###
        plt.plot(df,color='blue')
        plt.plot(prediction,color='red')
        plt.title("Original Data with Predictions for Two Years")
        plt.show()

        # Put the predictions into a .txt file
        txt = open("Q1_Daniel_March_24196320.txt",'w')
        txt.write('"x"')
        for line in prediction:
            txt.write("\n" + str(line))

    elif dataset == 2:
        df = pd.read_csv('q2_train.csv')
        df.Date = pd.to_datetime(df.Date)
        df.set_index("Date", inplace=True)

        df_diff = df - df.shift()
        df_diff.dropna(inplace=True)
        df_diff_2 = df - df.shift(52)
        df_diff_2.dropna(inplace=True)
        df_diff_3 = df_diff - df_diff.shift(52)
        df_diff_3.dropna(inplace=True)
        # plot_cf(df_diff,df_diff.activity)

        p,q = 0,2
        arma_fit = SARIMAX(df,order=(p,1,q),seasonal_order=(1,1,1,52)).fit(disp=-1)
        prediction = arma_fit.predict(start=525,end=525+103,dynamic=True)

        ## Plots the original data with the prediction ###
        plt.plot(df,color='blue')
        plt.plot(prediction,color='red')
        plt.title("Original Data with Predictions for Two Years")
        plt.show()

        # Put the predictions into a .txt file
        txt = open("Q2_Daniel_March_24196320.txt",'w')
        txt.write('"x"')
        for line in prediction:
            txt.write("\n" + str(line))

    if dataset == 3:
        df = pd.read_csv('q3_train.csv')
        df.Date = pd.to_datetime(df.Date)
        df.set_index("Date", inplace=True)

        ### Differenced Signal ###
        df_diff = df - df.shift()
        df_diff.dropna(inplace=True)
        df_diff_2 = df - df.shift(52)
        df_diff_2.dropna(inplace=True)
        df_diff_3 = df_diff - df_diff.shift(52)
        df_diff_3.dropna(inplace=True)

        p,q = 0,2
        arma_fit = SARIMAX(df,order=(p,1,q),seasonal_order=(1,1,1,52)).fit(disp=-1)
        prediction = arma_fit.predict(start=525,end=525+103,dynamic=True)

        ### Plots the original data with the prediction ###
        plt.plot(df,color='blue')
        plt.plot(prediction,color='red')
        plt.title("Original Data with Predictions for Two Years")
        plt.show()

        # Put the predictions into a .txt file
        txt = open("Q3_Daniel_March_24196320.txt",'w')
        txt.write('"x"')
        for line in prediction:
            txt.write("\n" + str(line))

    if dataset == 4:
        df = pd.read_csv('q4_train.csv')
        df.Date = pd.to_datetime(df.Date)
        df.set_index("Date", inplace=True)

        ### Differenced Signal ###
        df_diff = df - df.shift()
        df_diff.dropna(inplace=True)
        df_diff_2 = df - df.shift(52)
        df_diff_2.dropna(inplace=True)
        df_diff_3 = df_diff - df_diff.shift(52)
        df_diff_3.dropna(inplace=True)

        p,q = 0,2
        arma_fit = SARIMAX(df,order=(p,1,q),seasonal_order=(1,1,1,52)).fit(disp=-1)
        prediction = arma_fit.predict(start=525,end=525+103,dynamic=True)

        ### Plots the original data with the prediction ###
        plt.plot(df,color='blue')
        plt.plot(prediction,color='red')
        plt.title("Original Data with Predictions for Two Years")
        plt.show()

        # Put the predictions into a .txt file
        txt = open("Q4_Daniel_March_24196320.txt",'w')
        txt.write('"x"')
        for line in prediction:
            txt.write("\n" + str(line))

    elif dataset == 5:
        df = pd.read_csv('q5_train.csv')
        df.Date = pd.to_datetime(df.Date)
        df.set_index("Date", inplace=True)

        ### Differenced Signal ###
        df_diff = df - df.shift()
        df_diff.dropna(inplace=True)
        df_diff_2 = df - df.shift(52)
        df_diff_2.dropna(inplace=True)
        df_diff_3 = df_diff - df_diff.shift(52)
        df_diff_3.dropna(inplace=True)
        # plot_cf(df_diff,df_diff.activity)

        p,q = 0,2
        arma_fit = SARIMAX(df,order=(p,1,q),seasonal_order=(1,1,1,52)).fit(disp=-1)
        prediction = arma_fit.predict(start=525,end=525+103,dynamic=True)

        ### Plots the original data with the prediction ###
        plt.plot(df,color='blue')
        plt.plot(prediction,color='red')
        plt.title("Original Data with Predictions for Two Years")
        plt.show()

        # Put the predictions into a .txt file
        txt = open("Q5_Daniel_March_24196320.txt",'w')
        txt.write('"x"')
        for line in prediction:
            txt.write("\n" + str(line))
