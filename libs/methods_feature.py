# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:21:03 2020

@author: JY
"""
import inspect
from collections import Iterable

import numpy as np
import peakdetect
import ECDFtools
from scipy.stats import entropy


def mean(sig):
    return np.mean(sig)
def maximum(sig):
    return np.max(sig)
def minimum(sig):
    return np.min(sig)
def delta_maxmin(sig):
    return np.abs(np.abs(np.max(sig)) - np.abs(np.min(sig)))
def RMS(sig):
    return np.sqrt(np.square(sig).mean())
def variance(sig):
    return np.var(sig)
def fft_deviation(sig):
    fft_value = np.abs(np.fft.fft(sig))
    return fft_value.std()
    
# def entropy2(sig):
#     """ Computes entropy of label distribution. """
#     sig_int = sig.astype(np.int64)
#     n_labels = len(sig_int)
    
#     if n_labels < 2:
#         return 0
    
#     # c = Counter(sig_int)
#     counts = np.bincount(sig_int)
#     probs = counts / n_labels
#     n_classes = len(np.unique(sig_int))
    
#     if n_classes <= 1:
#         return 0

#     return entropy(probs, base=n_classes)
def entropy_e(sig):
    """ Computes entropy of label distribution. """
    # sig_int = sig.astype(np.int64)
    n_labels = len(sig)
    
    if n_labels < 2:
        return 0
    p = sig/sig.sum(keepdims=True)
    return  entropy(p)

def ECDF_representation_1d( sig, n=15 ):
    X = []
    sig = np.array(sig)
    tmp_ = sig + np.random.randn(len(sig))*0.01
    x = np.sort(tmp_)
    f = np.arange(1, len(x)+1)/float(len(x))
    
    ll = np.interp(np.linspace(0,1,n), x, f)
    X = ll
    return X

def maxNumConseExceed(sig, thre=500):
    """
    Out: Integer(number of point exceed 'thre' in 'sig')
    
    return maximum number of consecutive exceeded value
    """
    exceed_tf = np.array(sig)>thre
    
    true_num = np.diff(np.where(np.concatenate(([exceed_tf[0]],
                                                 exceed_tf[:-1] != exceed_tf[1:],
                                                 [True])))[0])[::2]
                                                
    if len(true_num)==0:
        res=0
    else:
        res=true_num[0]
    return res
    

def sidebyValue(sig, delta=500, prop=0.3, ignore_under=0.03):
    """
    In: Signal
    Out: Integer(side)
        -999: cannot specify
        -1: Minus side
         0: neutral
         1: Plus side
         
    Return the dominant side of signal 
    by comparing amounts of value exceed the minimum threshold(delta)
    """
    plus_exceed = np.where( sig > delta )[0]
    minus_exceed = np.where( sig < -delta )[0]
    
    plus_portion = len(plus_exceed)/len(sig)
    minus_portion = len(minus_exceed)/len(sig)
#    print(len(plus_exceed),len(minus_exceed))
    
    if plus_portion+minus_portion < ignore_under:
        return 0, plus_portion, minus_portion
    elif np.abs(plus_portion-minus_portion) > prop:
        if plus_portion > minus_portion:
            return 1, plus_portion, minus_portion
        else:
            return -1, plus_portion, minus_portion
    else:
        return -999, plus_portion, minus_portion
    

def sidebyPeak(sig,look_a=4, delta=500):
    """
    Out: Integer(side)
        -1: Minus side
         0: neutral(+cannot specify)
         1: Plus side
         3: Rubbing
         
    Return the dominant side of signal 
    by the number of peak occured and the biggest peak's absolute value
    if there are more than 4 peaks return as Rubbing
    """
    res = peakdetect.peakdetect(y_axis=sig, lookahead=look_a, delta=delta)
    num_plus = len(res[0])
    num_minus = len(res[1])
    
    if num_plus + num_minus > 4:
        return 3
    
    if num_plus==0 and num_minus==0:
        return 0
    elif num_plus>0 and num_minus==0:
        return 1
    elif num_plus==0 and num_minus>0:
        return -1
    else:
        max_index = np.argmax(np.array(res[0])[:,1])
        min_index = np.argmin(np.array(res[1])[:,1])
        
        max_val = res[0][max_index][1]
        min_val = res[1][min_index][1]
        
        if min_val >0:
            return 1
        elif max_val<0:
            return -1

        d_max = max_val-np.abs(min_val)
        if np.abs(d_max) < delta:
            max_x = res[0][max_index][0]
            min_x = res[1][min_index][0]
            if np.abs(max_x - min_x)<25:
                if max_x<min_x:
                    return 1
                else:
                    return -1
            return 0
        elif d_max > 0:
            return 1
        elif d_max < 0:
            return -1
        else:
            print("Plus/Minus max is exactely same value")
            return 0

def sidePeakNum(sig,look_a=4, delta=500):
    """
    Out: plus/minus number count
    """
    res = peakdetect.peakdetect(y_axis=sig, lookahead=look_a, delta=delta)
    num_plus = len(res[0])
    num_minus = len(res[1])
    
    return num_plus, num_minus

def totPeakNum(sig,look_a=4, delta=500):
    """
    Out: both peaks number count
    """
    res = peakdetect.peakdetect(y_axis=sig, lookahead=look_a, delta=delta)
    num_plus = len(res[0])
    num_minus = len(res[1])
    
    return num_plus+num_minus

def avgDistancePlusPeaks(sig, look_a=4, delta=1500):
    """
    Out: Float(average distance of plus peak)
    
    Retrun total peak occurences and average time among peaks 
    use GYRO_Z signal which is stable than EOG signal and precise enough to detect nose rubbing input
    """
    res = peakdetect.peakdetect(y_axis=sig, lookahead=look_a, delta=delta)

    if len(res[0])>0:
        plus_xs = np.array(res[0])[:,0]
        mean_dx = np.average(np.diff(plus_xs))
    else:
        mean_dx = -999
    return  mean_dx

# def getRubingInfo(sig, look_a=4, delta=1500):
#     """
#     Out: Integer(total number of peaks), Float(average distance of plus peak)
    
#     Retrun total peak occurences and average time among peaks 
#     use GYRO_Z signal which is stable than EOG signal and precise enough to detect nose rubbing input
#     """
#     res = peakdetect.peakdetect(y_axis=sig, lookahead=look_a, delta=delta)
#     num_plus = len(res[0])
#     num_minus = len(res[1])
    
    
#     plus_xs = np.array(res[0])[:,0]
#     mean_dx = np.average(np.diff(plus_xs))
    
#     return num_plus+num_minus, mean_dx

class sumAllFeatures:
    def __init__(self, target_methods=[]):
        self.target_methods = target_methods if len(target_methods)>0 else [a[0] for a in inspect.getmembers(sys.modules[__name__], inspect.isfunction) if not '__' in a[0]]
        self.f_label = []
    def cal_features_from_one_signal(self, sig):
        featureX = np.array([])
        for one_target in self.target_methods:
            tmp_result = getattr(sys.modules[__name__], one_target)(sig)
            featureX = np.append(featureX, tmp_result)
        return featureX
    def update_label(self):
        f_label = []
        sig = np.arange(100)
        for one_target in self.target_methods:
            tmp_result = getattr(sys.modules[__name__], one_target)(sig)
            
            if isinstance(tmp_result, Iterable):
                for ii, k in enumerate(tmp_result):
                    f_label.append("{}_{}".format(one_target, ii))
            else:
                # print(type(tmp_result))
                f_label.append(one_target)
        self.f_label = f_label
        return f_label
#%%
import sys
import matplotlib.pyplot as plt


if __name__ == "__main__":
    test_sum = sumAllFeatures()
    sig = 1500*np.cos(np.arange(100)*0.2)
    plt.plot(sig)
    
    features = test_sum.cal_features_from_one_signal(sig)
    # test_sum.update_label()
    labels = test_sum.update_label()
    print("Total features num: {}".format(len(labels)))
    print(labels)
    
