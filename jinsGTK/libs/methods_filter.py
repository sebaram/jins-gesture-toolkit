# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:21:24 2020

@author: JY
"""

from scipy import signal



def butter_lowpass_filter(sig, cutoff=7, fs=100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    y = signal.lfilter(b, a, sig)
    return y

