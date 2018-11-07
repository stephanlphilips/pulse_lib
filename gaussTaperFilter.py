# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:42:07 2018

@author: Administrator
"""
import numpy as np
import scipy.signal as signal

def gaussTaperFilter (waveform):
#    taps = np.genfromtxt('gaussTaper4.csv', delimiter = ',')
    taps = np.genfromtxt('gaussTaper4dTaps.csv', delimiter = ',')
    wave = np.concatenate([waveform[-len(taps):], waveform, waveform[:len(taps)]])
    filtered = signal.lfilter(taps,[1.0], wave)
    filtered = filtered[len(taps) : -len(taps)]
    # filtered *= 0.31
    return filtered

