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
df = pd.read_pickle("CollectedData/2020-04-28 10_03_39_P0_EXP1_segmented.pickle")


TARGET_MODEL = 'RDFclassifier'
TARGET_FILTER = ['butter_lowpass_filter']
# TARGET_FEATURES = ['ECDF_representation_1d', 'RMS', 'avgDistancePlusPeaks', 'delta_maxmin', 'entropy', 'entropy_e', 'fft_deviation', 'maxNumConseExceed', 'maximum', 'mean', 'minimum', 'sidePeakNum', 'sidebyPeak', 'sidebyValue', 'totPeakNum', 'variance']
TARGET_FEATURES = ['ECDF_representation_1d', 'RMS', 'avgDistancePlusPeaks',
                   'delta_maxmin', 'entropy_e', 'fft_deviation',
                   'maxNumConseExceed', 'maximum', 'mean', 
                   'minimum', 'sidePeakNum', 'sidebyPeak', 'sidebyValue',
                   'totPeakNum', 'variance']

TARGET_AXIS = ['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V', 
                'GYRO_X', 'GYRO_Y', 'GYRO_Z',
                'ACC_X', 'ACC_Y', 'ACC_Z']
# TARGET_AXIS = ['EOG_L', 'EOG_R']


test_sum = methods_feature.sumAllFeatures(TARGET_FEATURES)

totalX = []
totaly = []

for i,row in df.iterrows():
    if row.TrialNum <0:
        continue
    total_label = []
    rowX = np.array([])
    this_data = row.DATA.copy()
    for one_axis in TARGET_AXIS:
        sig = this_data[one_axis].values
        for one_filter in TARGET_FILTER:
            sig = getattr(methods_filter, one_filter)(sig)
            
        partX = test_sum.cal_features_from_one_signal(sig)
        
        rowX = np.append(rowX, partX)
        total_label += [one_axis+"_"+one_name for one_name in test_sum.update_label()]
    totalX.append(list(rowX))
    totaly.append(row.Target)
    
    # break
    
model = getattr(methods_model, TARGET_MODEL)()
results_ = model.get_confusion_matrix(totalX,totaly,cv=2)

print("mean:{:.2f}%, std:{:.2f}%".format(results_[0].mean()*100, results_[0].std()*100))
print(results_[1])
