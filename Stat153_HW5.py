import numpy as np
import scipy
from math import *
import matplotlib.pyplot as plt

def q2():
    """For question 2 of the assignment."""
    # Generate time domain points:
    f1 = lambda x: sin(pi*(x**2)/256)
    pts_1 = [f1(i) for i in range(128)]
    # DFT of the points:
    dft_subset_1 = np.fft.fft(pts)[:64]
    # Magnitudes of DFTs:
    mags_1 = [np.absolute(x) for x in dft_subset]

    # Same thing for sine function with 512 in denominator:
    f2 = lambda x: sin(pi*(x**2)/512)
    pts_2 = [f1(i) for i in range(128)]
    dft_subset_2 = np.fft.fft(pts)[:64]
    mags_2 = [np.absolute(x) for x in dft_subset]

    # Same thing for sine function with 1024 in denominator:
    f3 = lambda x: sin(pi*(x**2)/1024)
    pts_3 = [f1(i) for i in range(128)]
    dft_subset_3 = np.fft.fft(pts)[:64]
    mags_3 = [np.absolute(x) for x in dft_subset]

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
