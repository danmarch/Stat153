import numpy as np
import scipy as sp
from math import *
from cmath import exp as c_exp
import matplotlib.pyplot as plt
from pylab import *
from numpy.random import normal
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

def q2():
    "For question 2 of the assignment."
    # Generate time domain points:
    f1 = lambda x: sin(pi*(x**2)/256)
    pts_1 = [f1(i) for i in range(128)]
    # DFT of the points:
    dft_subset_1 = np.fft.fft(pts_1)[:64]
    # Magnitudes of DFTs:
    mags_1 = [np.absolute(x) for x in dft_subset_1]

    # Same thing for sine function with 512 in denominator:
    f2 = lambda x: sin(pi*(x**2)/512)
    pts_2 = [f2(i) for i in range(128)]
    dft_subset_2 = np.fft.fft(pts_2)[:64]
    mags_2 = [np.absolute(x) for x in dft_subset_2]

    # Same thing for sine function with 1024 in denominator:
    f3 = lambda x: sin(pi*(x**2)/1024)
    pts_3 = [f3(i) for i in range(128)]
    dft_subset_3 = np.fft.fft(pts_3)[:64]
    mags_3 = [np.absolute(x) for x in dft_subset_3]

    plt.plot(pts_1)
    plt.title("Time Domain Plot")
    plt.show()
    plt.plot(mags_1)
    plt.title("Frequency Domain Magnitudes")
    plt.show()
    plt.plot(pts_2)
    plt.title("Time Domain Plot")
    plt.show()
    plt.plot(mags_2)
    plt.title("Frequency Domain Magnitudes")
    plt.show()
    plt.plot(pts_3)
    plt.title("Time Domain Plot")
    plt.show()
    plt.plot(mags_3)
    plt.title("Frequency Domain Magnitudes")
    plt.show()

def q4():
    "For question 4 of the assignment."
    domain = np.linspace(-.5,.5,128)
    f1 = lambda x: (1-0.9*c_exp(-24j*pi*x)-0.5*c_exp(-2j*pi*x)+\
                   0.045*c_exp(-26j*pi*x))
    f2 = lambda x: (1-0.5*c_exp(-2j*pi*x))
    f3 = lambda x: (1-0.9*c_exp(-4j*pi*x))
    g1 = lambda y: 1 / (np.absolute(f1(y))**2)
    g2 = lambda y: 1 / (np.absolute(f2(y))**2)
    g3 = lambda y: 1 / (np.absolute(f3(y))**2)
    pts1 = [g1(i) for i in domain]
    pts2 = [g2(i) for i in domain]
    pts3 = [g3(i) for i in domain]
    plt.plot(domain,pts1)
    plt.title("Spectral Density Q4PB")
    plt.show()
    plt.plot(domain,pts2)
    plt.title("Spectral Density Q4PC, Part 1")
    plt.show()
    plt.plot(domain,pts3)
    plt.title("Spectral Density Q4PC, Part 2")
    plt.show()

def q6():
    "For question 6 of the assignment."
    domain = np.linspace(-.5,.5,128)
    f1 = lambda x: (1-0.99*c_exp(-6j*pi*x))
    g1 = lambda y: 1 / (np.absolute(f1(y))**2)
    pts1 = [g1(i) for i in domain]
    plt.plot(domain,pts1)
    plt.title("Spectral Density Q6PA")
    plt.show()
    helper_func = lambda x: (2*(sin(2*pi*x)*(1/tan(pi*x)))-1)/3
    trans_func = lambda y: helper_func(y)**2
    f2 = lambda x: g1(x) * trans_func(x)
    pts2 = [f2(i) for i in domain]
    plt.plot(domain,pts2)
    plt.title("Spectral Density Q6PD")
    plt.show()

    df = pd.DataFrame(pts1)
    new_data = pd.rolling_mean(df,window=3)
    plt.plot(domain,new_data)
    plt.show()

def _q7():
    coef = [-.5,0,0,0,0,0,0,0,0,0,0,-.7,.35]
    sigma = 1
    n = 400
    return q7(coef,sigma,n)

def q7(coef,sigma,n):
    """Implementation of an AR process sampling.
    @param coef: The list of AR coefficients.
    @param sigma: The variance of the white noise distribution.
    @param n: The number of data points to be generated.
    @return return_array: A n-length list of data points.
    """
    distribution = normal(0,sigma,n)
    return_array = array([])
    val = 0.0
    length = len(coef)
    for i in range(n):
        if i < length:
            return_array = append(return_array, distribution[i])
        else:
            val = 0.0
            for j in range(length):
                val += coef[j] * return_array[i-j-1]
            return_array = append(return_array, val + distribution[i])
    lag_acf = acf(return_array, nlags=20)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(return_array)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(return_array)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    plt.show()
    return return_array

def q8pA():
    "For question 8 of the assignment."
    df = pd.read_csv('beer.csv')
    length = float(len(df))
    df.Week = pd.to_datetime(df.Week)
    df.set_index("Week", inplace=True)
    seasonal_diff = df - df.shift(52)
    final_diff = seasonal_diff - seasonal_diff.shift(1)
    dft_vals = np.fft.fft(final_diff)
    periodogram = np.array([np.absolute(i)**2 for i in dft_vals])
    periodogram = np.divide(periodogram,length)
    moving_avg = pd.rolling_mean(periodogram,window=7)
    plt.plot(moving_avg)
    plt.title("Non-Parametrically Approximated Spectral Density, Q8PA")
    plt.show()
