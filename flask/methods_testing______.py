# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:43:10 2020

@author: kctm
"""
import sys
sys.path.append("../libs")

import methods_filter, methods_model, methods_feature


#%%
import numpy as np
import pandas as pd
from scipy import signal

df = pd.read_pickle("CollectedData/2020-04-28 10_03_39_P0_EXP1_segmented.pickle")

df = pd.read_pickle("CollectedData/2020-05-02 19_12_29_P0_EXP1_segmented.pickle")

TARGET_MODEL = 'RDFclassifier'
TARGET_FILTER = ['butter_lowpass_filter']
# TARGET_FEATURES = ['ECDF_representation_1d', 'RMS', 'avgDistancePlusPeaks', 'delta_maxmin', 'entropy', 'entropy_e', 'fft_deviation', 'maxNumConseExceed', 'maximum', 'mean', 'minimum', 'sidePeakNum', 'sidebyPeak', 'sidebyValue', 'totPeakNum', 'variance']

# TARGET_RAW_AXIS = ['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V', 
#                 'GYRO_X', 'GYRO_Y', 'GYRO_Z',
#                 'ACC_X', 'ACC_Y', 'ACC_Z']
TARGET_RAW_AXIS = ['EOG', 'EOG_fft', 'ACC', 'ACC_fft', 'GYRO', 'GYRO_fft']

TARGET_FEATURE_AXIS = ['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V', 
                'GYRO_X', 'GYRO_Y', 'GYRO_Z',
                'ACC_X', 'ACC_Y', 'ACC_Z']
TARGET_FEATURE_AXIS = ['EOG', 'EOG_diff', 'ACC', 'ACC_diff', 'GYRO', 'GYRO_diff']
# TARGET_AXIS = ['EOG_L', 'EOG_R']
TARGET_FEATURES = ['ECDF_representation_1d', 'RMS', 'avgDistancePlusPeaks',
                   'delta_maxmin', 'entropy_e', 'fft_deviation',
                   'maxNumConseExceed', 'maximum', 'mean', 
                   'minimum', 'sidePeakNum', 'sidebyPeak', 'sidebyValue',
                   'totPeakNum', 'variance']


def get_train_result(target_df, TARGET_MODEL, TARGET_FILTER, TARGET_RAW_AXIS,TARGET_FEATURE_AXIS,TARGET_FEATURES):
    test_sum = methods_feature.sumAllFeatures(TARGET_FEATURES)
    
    totalX = []
    totaly = []
    
    AXIS_SET = {'EOG': ['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V'],
                'ACC':  ['ACC_X', 'ACC_Y', 'ACC_Z'],
                'GYRO': ['GYRO_X', 'GYRO_Y', 'GYRO_Z']}
    W_LENGTH = 50
    
    for i,row in target_df.iterrows():
        if row.TrialNum <0:
            continue
        total_label = []
        rowX = np.array([])
        this_data = row.DATA.copy()
        
        
        for one_axis_set in TARGET_RAW_AXIS:
            if '_fft' in one_axis_set:
                target_set = one_axis_set.replace('_fft','')
                is_fft = True
            else:
                target_set = one_axis_set
                is_fft = False
            
            for one_part in AXIS_SET[target_set]:
                    tmp_raw = signal.resample(this_data[one_part].values, W_LENGTH)
                    if is_fft:
                        tmp_raw = np.abs(np.fft.fft(tmp_raw))
                        post_fix = 'fft'
                    else:
                        post_fix = ''
                    rowX = np.append(rowX, tmp_raw)
                    total_label += [one_part+"{}_{:03d}".format(post_fix,i) for i in range(W_LENGTH)]
                    
            
        for one_axis in TARGET_FEATURE_AXIS:
            if 'diff' in one_axis:
                target_set = one_axis.replace('_diff','')
                is_diff = True
            else:
                target_set
                is_diff = False
                
            for one_part in AXIS_SET[target_set]:
                sig = this_data[one_part].values
                if is_diff:
                    sig = np.diff(sig)
                
                for one_filter in TARGET_FILTER:
                    sig = getattr(methods_filter, one_filter)(sig)
                
                partX = test_sum.cal_features_from_one_signal(sig)
            
                rowX = np.append(rowX, partX)
                total_label += [one_axis+"_"+one_name for one_name in test_sum.update_label()]
                
        totalX.append(list(rowX))
        totaly.append(row.Target)
        
        
    model = getattr(methods_model, TARGET_MODEL)()
    results_ = model.get_confusion_matrix(totalX,totaly,cv=2)
    
    result_str = "Model: {}".format(TARGET_MODEL)+"\n"\
                    +"mean:{:.2f}%, std:{:.2f}%".format(results_[0].mean()*100, results_[0].std()*100)+"\n"\
                    +str(results_[1])
    print(result_str)
    return result_str


result_str = get_train_result(df, TARGET_MODEL, TARGET_FILTER, TARGET_RAW_AXIS,TARGET_FEATURE_AXIS,TARGET_FEATURES)