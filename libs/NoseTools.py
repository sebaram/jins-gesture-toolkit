# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import time, os
try:
    import peakdetect
    import ECDFtools
except:
    import library.peakdetect as peakdetect
    import library.ECDFtools as ECDFtools

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score


import matplotlib.pyplot as plt
        

left_rightString = {1: 'Left',
                    0: 'nan',
                    2: 'Right',
                    3: 'Rub'}
bounce_pushString = {0: 'nan',
                     1: 'Flick',
                     2: 'Push',
                     3: ''}
TYPE_STR = {-1: "Nothing",
            0: "Left Flick",
            1: "Left Push",
            2: "Right Flick",
            3: "Right Push",
            4: "Rubbing"}
                     
current_milli_time = lambda: int(round(time.time() * 1000))
current_milli_time_f = lambda: float(time.time() * 1000)

def consecutive(data, stepsize=0):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    
def translateResult(data):
    string = "error"
    if len(data)==6:
        string = left_rightString[data[0]] +"|"+ left_rightString[data[1]] +"|"+ bounce_pushString[data[2]]+"|"+ str(data[3]) +"|"+ str(data[4])+"|"+str(data[5])
    elif len(data)==5:
        string = left_rightString[data[0]] +"|"+ bounce_pushString[data[1]] +"|"+ str(data[2]) +"|"+ str(data[3])+"|"+str(data[4])
    return string

def allSame(data):
    tf = data==data[0]
    res = len(data)==np.sum(tf)
    return res
    
TYPE_RESULT = {-1: np.array([0,0,-1,0]),
            0: np.array([1,1,0,0]),
            1: np.array([1,2,1,0]),
            2: np.array([2,1,2,0]),
            3: np.array([2,2,3,0]),
            4: np.array([3,3,4,0])}
                        
class plotEOG:
    def __init__(self, plot_num=2, val_num=[2,2], val_name=[['EOG_H','EOG_V'],['EOG_L','EOG_R']]):
        self.plot_num = plot_num
        self.val_num = val_num
        self.val_name = val_name
        
        self.HV_LR = ["HV","LR"]
        
        self.i = 0
        self.w_size = 100
        
        self.fig = [0 for i in range(self.plot_num)]
        self.ax = [0 for i in range(self.plot_num)]
        self.lines = dict()
        
                      
    def initPlot(self, vals):
#        if len(vals)!=self.plot_num:
#            print("Plot number dont match")
#            return 0
#        for i,inner_val in enumerate(vals):
#            if len(inner_val)!=self.val_num[i]:
#                print("Value for plot dont match:%f"%i)
#                return 0
            
            
        for i in range(self.plot_num):
            self.fig[i], self.ax[i] = plt.subplots()
            for k in range(self.val_num[i]):
                self.lines[self.val_name[i][k]], = self.ax[i].plot(vals[self.val_name[i][k]], label=self.val_name[i][k])
                
#            self.ax[i].set_xlim([2,1+self.w_size])
            self.ax[i].set_ylim([-3000,3000])
            self.ax[i].set_ylabel('EOG')
            self.ax[i].legend()
            
            
    def changeView(self, i):
        self.i = i
        
        for ax in self.ax:
            ax.set_xlim([self.i, self.i+self.w_size])   
            
    def changeData(self, vals):
        for key in self.lines.keys():
            try:
                self.lines[key].set_data(vals['TIME'],vals[key])
            except:
                xs = range(len(vals[key]))
                self.lines[key].set_data(xs,vals[key])  
        for ax in self.ax:
            ax.relim()
            ax.autoscale_view()
    def changeResultTitle(self, result_title):
        for fig in self.fig:
            fig.suptitle(result_title, fontsize=14, fontweight='bold')
    def saveFig(self, name, printing=True):
        for i in range(self.plot_num):
            self.fig[i].savefig(name+'_%s.png'%(self.HV_LR[i]))
        if printing:
            print("saved: %s"%name)
            
    def close(self):
        for fig in self.fig:
            plt.close(fig)

class plotEOGinOne:
    def __init__(self, plot_num=2, val_num=[2,2], val_name=[['EOG_H','EOG_V'],['EOG_L','EOG_R']]):
        self.plot_num = plot_num
        self.val_num = val_num
        self.val_name = val_name
        
        self.HV_LR = ["HV","LR"]
        
        self.i = 0
        self.w_size = 100
        
        self.fig = [0 for i in range(self.plot_num)]
        self.ax = [0 for i in range(self.plot_num)]
        self.lines = dict()
        
        
                      
    def initPlot(self, vals, y_max=3000, y_min=False):
#        if len(vals)!=self.plot_num:
#            print("Plot number dont match")
#            return 0
#        for i,inner_val in enumerate(vals):
#            if len(inner_val)!=self.val_num[i]:
#                print("Value for plot dont match:%f"%i)
#                return 0
        if y_min is False:
            y_min = -y_max
        self.fig, self.ax = plt.subplots(self.plot_num, sharex=True)
        self.fig.set_figheight(4*self.plot_num)
        self.fig.set_figwidth(5)
        for i in range(self.plot_num):
            for k in range(self.val_num[i]):
                self.lines[self.val_name[i][k]], = self.ax[i].plot(vals[self.val_name[i][k]], label=self.val_name[i][k])
                
#            self.ax[i].set_xlim([2,1+self.w_size])
            self.ax[i].set_ylim([y_min,y_max])
            self.ax[i].set_ylabel(str(self.val_name[i]))
            self.ax[i].legend()
            
        self.marker = [0 for i in range(self.plot_num)]
    def setMarker(self, plot_i, x, y, marker_size=100, color="r"):
        if self.marker[plot_i]==0:
            self.marker[plot_i] = self.ax[plot_i].scatter(x,y,marker='x',s=marker_size, c=color)
        else:
            self.marker[plot_i].set_offsets( [x,y] )
            self.marker[plot_i].set_color(color)
#            self.marker[plot_i]._sizes=marker_size
                
            
    def changeView(self, i):
        self.i = i
        
        for ax in self.ax:
            ax.set_xlim([self.i, self.i+self.w_size])   
            
    def changeData(self, vals):
        for key in self.lines.keys():
            
            try:
                self.lines[key].set_data(vals['TIME'],vals[key])
            except:
                xs = range(len(vals[key]))
                self.lines[key].set_data(xs,vals[key])  
        for ax in self.ax:
            ax.relim()
            ax.autoscale_view()
            
    def changeResultTitle(self, result_title):
        self.fig.suptitle(result_title, fontsize=14, fontweight='bold')
    def saveFig(self, name, printing=True):
        self.fig.savefig(name+'.png', dpi=100, bbox_inches='tight')
        
        if printing:
            print("saved: %s"%name)
            
    def close(self):
        plt.close(self.fig)
                
    
    
class RDFclassifier:
    def __init__(self, dt_ms, jins_client, sampling_dt=1000, stab_time=600, enable_graph = False, enable_fft_graph = False, f_TF='Forest_TF.pkl', f_Type='Forest_Type.pkl',
                 include_false_classification= True, target_thre_noFalse=0.50, others_thre_noFalse=0.25, target_thre_wFalse=0.30, others_thre_wFalse=0.13):
        self.jins_client = jins_client
        
        self.dt_ms = dt_ms
        self.last_t = current_milli_time()
        self.sampling_dt = sampling_dt
        
        # CODE for loading Trees
        self.forest_TF = joblib.load(f_TF) 
        self.forest_Type = joblib.load(f_Type)
        
        self.store_n = 10
        self.store_Thre = 0.7
        self.stab_time = stab_time
#==============================================================================
#         self.last_res = np.zeros((self.store_n, 4))
#         self.last_prop = np.zeros((self.store_n, 6))
#         self.last_time = np.zeros(self.store_n)
#==============================================================================
        self.resetStoringArray()
        self.last_activationTime = 0
        
        self.enable_graph = enable_graph
        self.enable_fft_graph = enable_fft_graph
        
        if self.enable_graph:
            
#            self.plot_graph  = plotEOG()
#            vals_dict = {"GYRO_X": np.random.randint(500, size=100),
#                        "GYRO_Y": np.random.randint(500, size=100),
#                        "GYRO_Z": np.random.randint(500, size=100),
#                        "EOG_L": np.random.randint(500, size=100),
#                        "EOG_R": np.random.randint(500, size=100),
#                        "EOG_H": np.random.randint(500, size=100),
#                        "EOG_V": np.random.randint(500, size=100)
#                            }
#            self.plot_graph.initPlot(vals_dict)
            self.plot_graph = plotEOGinOne(plot_num=4, val_num=[2,2,3,3], val_name=[['EOG_H','EOG_V'],['EOG_L','EOG_R'],["ACC_X" ,"ACC_Y" ,"ACC_Z"],["GYRO_X" ,"GYRO_Y" ,"GYRO_Z"]])
            vals_dict = {"ACC_X": np.random.randint(500, size=100),
                        "ACC_Y": np.random.randint(500, size=100),
                        "ACC_Z": np.random.randint(500, size=100),
                        
                        "GYRO_X": np.random.randint(500, size=100),
                        "GYRO_Y": np.random.randint(500, size=100),
                        "GYRO_Z": np.random.randint(500, size=100),
                        
                        "EOG_L": np.random.randint(500, size=100),
                        "EOG_R": np.random.randint(500, size=100),
                        "EOG_H": np.random.randint(500, size=100),
                        "EOG_V": np.random.randint(500, size=100)
                            }
            self.plot_graph.initPlot(vals_dict)
        
        if self.enable_fft_graph:
            self.plot_fft = plotEOGinOne(plot_num=2, val_num=[3, 1], val_name=[['FFT_X','FFT_Y','FFT_Z'],['FFT_SUM']])
            vals_dict = {                        
                        "FFT_X": np.random.randint(500, size=100),
                        "FFT_Y": np.random.randint(500, size=100),
                        "FFT_Z": np.random.randint(500, size=100),
                        
                        "FFT_SUM": np.random.randint(500, size=100)
                            }
            self.plot_fft.initPlot(vals_dict, y_max=700000, y_min=0)
            
        self.include_false_classification = include_false_classification
        self.target_thre_noFalse = target_thre_noFalse
        self.others_thre_noFalse = others_thre_noFalse
        
        self.target_thre_wFalse = target_thre_wFalse
        self.others_thre_wFalse = others_thre_wFalse
        
        self.TYPE_STR = {  -1: "Nothing",
                            0: "Left Flick",
                            1: "Left Push",
                            2: "Right Flick",
                            3: "Right Push",
                            4: "Rubbing"}
        
    def resetStoringArray(self):
        self.last_prop = np.zeros((self.store_n, self.forest_Type.n_classes_))
        self.last_res = np.zeros((self.store_n))
        self.last_time = np.zeros(self.store_n)
        
    def runJinsclient(self, cur_t, printing=False):
        if cur_t-self.last_t < self.dt_ms:
            return self.last_res[-1], self.last_prop[-1,:]
            
#        data = self.jins_client.getLast(100) #time, l, r, g_x, g_y, g_z
        data = self.jins_client.getLastbyTime_dict(dt=self.sampling_dt,delta_value=True)
        
        res, res_prop = self.calculateThisData(data, printing=printing)
        
        self.last_t = current_milli_time()
        
        self.last_res = np.roll(self.last_res, -1, axis=0)
        self.last_res[-1] = res
        
        self.last_prop = np.roll(self.last_prop, -1, axis=0)
        self.last_prop[-1,:] = res_prop
        
        self.last_time = np.roll(self.last_time, -1, axis=0)
        self.last_time[-1] = cur_t
        
        return self.last_res[-1], self.last_prop[-1,:]
        
    def runJinsTypeonly(self, cur_t, printing=False):
        if cur_t-self.last_t < self.dt_ms:
            return self.last_res[-1], self.last_prop[-1,:]
            
        
#        data = self.jins_client.getLast_dict(100) 
        data = self.jins_client.getLastbyTime_dict(dt=self.sampling_dt,delta_value=True)
#        print("runJinsTypeonly: ",data.keys())
        res, res_prop = self.calculateJustType(data, printing=printing)
        
        self.last_t = current_milli_time()
        
        self.last_res = np.roll(self.last_res, -1, axis=0)
        self.last_res[-1] = res
        
        self.last_prop = np.roll(self.last_prop, -1, axis=0)
        self.last_prop[-1,:] = res_prop
        
        self.last_time = np.roll(self.last_time, -1, axis=0)
        self.last_time[-1] = cur_t
        
        return self.last_res[-1], self.last_prop[-1,:]
        
    def runManualTypeonly(self, cur_t, data, printing=False):
        
        res, res_prop = self.calculateJustType(data, printing=printing)
        
        self.last_res = np.roll(self.last_res, -1, axis=0)
        self.last_res[-1] = res
        
        self.last_prop = np.roll(self.last_prop, -1, axis=0)
        self.last_prop[-1,:] = res_prop
        
        self.last_time = np.roll(self.last_time, -1, axis=0)
        self.last_time[-1] = cur_t
        
        return self.last_res[-1], self.last_prop[-1,:]
        
    def calculateThisData(self, data, printing=False):
       
        input_data = makeX_wDict(data)
        
        result_Type_pro = np.zeros(5, dtype=int)
        
        result_TF = self.forest_TF.predict_proba(input_data)
        result_TF = result_TF[0]
        if printing: print("RDFclassifier| TF Porb.: ",result_TF)
        
        if result_TF[1]>result_TF[0] and result_TF[1]>self.target_thre_noFalse:
            result_Type_pro = self.forest_Type.predict_proba(list(input_data))
            result_Type_pro = result_Type_pro[0]
            
            d_pro = result_Type_pro.max()-result_Type_pro
            if result_Type_pro.max()>self.target_thre_noFalse or d_pro.min()>self.others_thre_noFalse:
                result_Type = np.argmax(result_Type_pro)
            else:
                result_Type = -1
            
            if printing: print("RDFclassifier| Type Porb.: ",self.TYPE_STR[result_Type], "|",result_Type_pro)
            
            
#            if result_Type in [0,1,2,3,4]:
#                res = TYPE_RESULT[result_Type]
#            else:
#                res = TYPE_RESULT[-1]
            
        else:
            result_Type = -1
            
        
        if printing:
            print("RDFclassifier| Result:",result_Type)
        if self.enable_graph:
            self.plot_graph.changeData(data)
            self.plot_graph.changeResultTitle(self.TYPE_STR[result_Type]+"|%d"%(len(data['TIME'])))
        if self.enable_fft_graph:
            half_size = int(len(data['GYRO_X'])/2)
            x_val=np.linspace(0,50,half_size)
            fftX = np.abs(np.fft.fft(data['GYRO_X'])[:half_size])
            fftY = np.abs(np.fft.fft(data['GYRO_Y'])[:half_size])
            fftZ = np.abs(np.fft.fft(data['GYRO_Z'])[:half_size])
            fft_data = { "FFT_X": fftX,
                         "FFT_Y": fftY,
                         "FFT_Z": fftZ,
                        
                         "FFT_SUM": fftX+fftY+fftZ,
                         
                         "TIME": x_val}
            self.plot_fft.changeData(fft_data)
            self.plot_fft.changeResultTitle(self.TYPE_STR[result_Type]+"|%d"%(len(data['TIME'])))
            
            self.plot_fft.setMarker(1,np.random.randint(50),np.random.randint(600000))
        return result_Type, result_Type_pro    
        
    def calculateJustType(self, data, printing=False):

        input_data = makeX_wDict(data)
        
        result_Type_pro = np.zeros(5, dtype=int)
        
        input_data = np.array([input_data])
#        print("runJustType", input_data.shape)
        result_Type_pro = self.forest_Type.predict_proba(input_data)
        result_Type_pro = result_Type_pro[0]
        
        d_pro = result_Type_pro[1:].max()-result_Type_pro[1:]
        
        
        if self.include_false_classification:
            if result_Type_pro[1:].max()>self.target_thre_wFalse or d_pro.min()>self.others_thre_wFalse:
                result_Type = np.argmax(result_Type_pro[1:])
            else:
                result_Type = -1
        else:
            if result_Type_pro[0:].max()>self.target_thre_noFalse or d_pro.min()>self.others_thre_noFalse:
                result_Type = np.argmax(result_Type_pro[0:])
            else:
                result_Type = -1
        
        
            
        
        if printing: print("RDFclassifier| Type Porb.: ",self.TYPE_STR[result_Type], "|",result_Type_pro)
        
            
        if printing:
            print("RDFclassifier| Result:",result_Type)
        if self.enable_graph:
            self.plot_graph.changeData(data)
            self.plot_graph.changeResultTitle(self.TYPE_STR[result_Type]+"|%d"%(len(data['TIME'])))
        if self.enable_fft_graph:
            half_size = int(len(data['GYRO_X'])/2)
            x_val=np.linspace(0,50,half_size)
            fftX = np.abs(np.fft.fft(data['GYRO_X'])[:half_size])
            fftY = np.abs(np.fft.fft(data['GYRO_Y'])[:half_size])
            fftZ = np.abs(np.fft.fft(data['GYRO_Z'])[:half_size])
            fft_data = { "FFT_X": fftX[:20],
                         "FFT_Y": fftY[:20],
                         "FFT_Z": fftZ[:20],
                        
                         "FFT_SUM": (fftX+fftY+fftZ)[:20],
                         
                         "TIME": x_val[:20]}
            self.plot_fft.changeData(fft_data)
            
            
            
            max_val = fft_data["FFT_SUM"].max()
            sum_val = fft_data["FFT_SUM"].sum()
            max_freq = np.where(fft_data["FFT_SUM"]>=max_val)[0]
            if sum_val!=0 and (max_val/sum_val) > 0.15 and result_Type==4:           
                self.plot_fft.setMarker(1,max_freq,max_val)
            else:
                self.plot_fft.setMarker(1,-10,-10)
                
            if result_Type==4:
                self.plot_fft.changeResultTitle(self.TYPE_STR[result_Type]+"|%.2f|%d"%(max_freq[0]*x_val[1],max_val))
            else:
                self.plot_fft.changeResultTitle("-|%.2f|%d"%(max_freq[0]*x_val[1],max_val))

        return result_Type, result_Type_pro
        
    def getStabResult(self):
        if current_milli_time() - self.last_activationTime < self.sampling_dt:
            return -1, self.last_prop[-1,:]
        
        picked_res_index = np.where(self.last_time>(current_milli_time()-300))[0]
        if len(picked_res_index)==0:
            return -1, self.last_prop[-1,:]
        
        
        results = self.last_res[picked_res_index].astype(int)
        
        
        unique, counts = np.unique(results, return_counts=True)
        most_type = np.argmax(counts)
        
        if counts[most_type] > len(picked_res_index)*self.store_Thre:
            if unique[most_type] != -1:
                self.last_activationTime = current_milli_time()
            return unique[most_type], self.last_prop[-1,:]
        else:
            return -1, self.last_prop[-1,:]
    
    def close(self):
        if self.enable_graph:
            self.plot_graph.close()
        if self.enable_fft_graph:
            self.plot_fft.close()

        


        
"""
################################
Make input for tree(X)
################################
"""    

TRAIN_VALS = ["GYRO_X" ,"GYRO_Y" ,"GYRO_Z" ,"EOG_L" ,"EOG_R" ,"EOG_H" ,"EOG_V"]
#TRAIN_VALS = ["EOG_L" ,"EOG_R" ,"EOG_H" ,"EOG_V"]
ACCs = ["ACC_X" ,"ACC_Y" ,"ACC_Z"]
GYROs = ["GYRO_X" ,"GYRO_Y" ,"GYRO_Z"]
def makeX_wDict(data_dict: dict):
    cur_x = np.array([])
    p_m_nums = 0
#    print("makeX_wDict: ",data_dict.keys())
    for val in TRAIN_VALS:  #val is datatype:  ["ACC_X" ,"ACC_Y" ,"ACC_Z" ,"GYRO_X" ,"GYRO_Y" ,"GYRO_Z" ,"EOG_L" ,"EOG_R" ,"EOG_H" ,"EOG_V"]
        part_res =  useThisFeatures(data_dict[val])
        cur_x = np.append(cur_x, part_res)
        
        if val in ["EOG_L" ,"EOG_R"]:
            if type(p_m_nums) is int:
                p_m_nums = part_res[-2:]
            else:
                d_p_m_nums = np.abs(p_m_nums-part_res[-2:])
                
    cur_x = np.append(cur_x, d_p_m_nums)      
#    fft_sum = np.abs(np.fft.fft(data_dict["EOG_L"])) + np.abs(np.fft.fft(data_dict["EOG_R"]))
#    cur_x = np.append(cur_x, fft_sum)

    acc_rms = 0
    for val in ACCs:
        acc_rms += np.sqrt(np.square(data_dict[val]).mean())
    gyro_rms = 0
    for val in GYROs:
        gyro_rms += np.sqrt(np.square(data_dict[val]).mean())
      
    cur_x = np.append(cur_x, [acc_rms, gyro_rms])
    
    eog_rms = getEOG_RMS(data_dict)
    cur_x = np.append(cur_x, eog_rms)
#    print(len(cur_x))
    
    return cur_x

"""
earbit features
https://dl.acm.org/citation.cfm?id=3130902
"""

def makeX_wDict_earBit(data_dict):
    cur_x = np.array([])
    for one in GYROs:
        part_res = featuresEARBIT(data_dict[one])
        cur_x = np.append(cur_x, part_res)
	
    return cur_x
    



"""
Change this part to try other features
"""
COMMON_FEATURES = ["RMS","FFTdev","maxNumConse","plus_p100","minus_p100","plusPeak", "minusPeak"]
ADD_FEATURES = ["accRMS", "gyroRMS"]
def useThisFeatures(arr):
    # return features_170531(arr)
	return features_171220_weka(arr)
def featuresEARBIT(arr):
	"""Magnitude of motion."""
	rms = np.sqrt(np.square(part_x).mean())
	var = np.var(part_x)
	ent = entropy2(arr)
	fft_values = np.abs(np.fft.fft(arr))
	peak_power = np.max(fft_values)
	psd = np.sum(fft_values)
	
	"""Periodicity of motion"""
	
	
	
	
	
	

def features_170531(arr):
    rms = np.sqrt(np.square(arr).mean())
    num_exceed = maxNumConseExceed(arr)
    dummy, plus_portion, minu_portion = sidebyValue(arr)
    p_num, m_num = sidePeakNum(arr)
    
    out_arr = np.array([ rms, fft_deviation(arr), num_exceed, plus_portion, minu_portion, p_num, m_num])
    return out_arr 

def features_171220_weka(arr):
    rms = np.sqrt(np.square(arr).mean())
    num_exceed = maxNumConseExceed(arr)
    dummy, plus_portion, minu_portion = sidebyValue(arr)
    p_num, m_num = sidePeakNum(arr)
    
    out_arr = np.array([ np.int(rms), np.int(fft_deviation(arr)), num_exceed, plus_portion*100, minu_portion*100, p_num, m_num])
    return out_arr     
def features_17620wECDF(arr):
    rms = np.sqrt(np.square(arr).mean())
    num_exceed = maxNumConseExceed(arr)
    dummy, plus_portion, minu_portion = sidebyValue(arr)
    p_num, m_num = sidePeakNum(arr)
    
    ecdf = ECDFtools.ECDF_representation_1d(arr,15)
    
    
    out_arr = np.array([ rms, fft_deviation(arr), num_exceed, plus_portion, minu_portion, p_num, m_num])
    out_arr = np.concatenate((ecdf, out_arr))
    return out_arr 
    
def justOtherInfo(arr):
    rms = np.sqrt(np.square(arr).mean())
    num_exceed = maxNumConseExceed(arr)
    dummy, plus_portion, minu_portion = sidebyValue(arr)
    p_num, m_num = sidePeakNum(arr)
    
    out_arr = np.array([rms, arr.mean(), arr.std(), arr.max(), arr.min(), num_exceed, plus_portion, minu_portion, p_num, m_num])
    return out_arr        
        
def justOtherInfo_Entropy(arr):
    rms = np.sqrt(np.square(arr).mean())
    ent = entropy2(arr)
    num_exceed = maxNumConseExceed(arr)
    dummy, plus_portion, minu_portion = sidebyValue(arr)
    p_num, m_num = sidePeakNum(arr)
    
    out_arr = np.array([rms, ent, arr.mean(), arr.std(), arr.max(), arr.min(), num_exceed, plus_portion, minu_portion, p_num, m_num])
    return out_arr

def justOtherInfo_Entropy_FFTdev(arr):
    rms = np.sqrt(np.square(arr).mean())
    ent = entropy2(arr)
    num_exceed = maxNumConseExceed(arr)
    dummy, plus_portion, minu_portion = sidebyValue(arr)
    p_num, m_num = sidePeakNum(arr)
    
    out_arr = np.array([rms, ent, arr.mean(), arr.std(), arr.max(), arr.min(), fft_deviation(arr), num_exceed, plus_portion, minu_portion, p_num, m_num])
    return out_arr   
    

"""
################################
Feature Functions
################################
"""
def getEOG_RMS(data_dict):
    
    lr_rms = 0
    for val in ['EOG_L', 'EOG_R']:
        data = data_dict[val]
        if type(data)==np.ndarray:
            lr_rms += np.sqrt(np.square(data).mean())
        elif type(data)==pd.core.series.Series:
            lr_rms += np.sqrt(np.square(data.as_matrix()[0]).mean())
    hv_rms = 0
    for val in ['EOG_H', 'EOG_V']:
        data = data_dict[val]
        if type(data)==np.ndarray:
            hv_rms += np.sqrt(np.square(data).mean())
        elif type(data)==pd.core.series.Series:
            hv_rms += np.sqrt(np.square(data.as_matrix()[0]).mean())
    
    return [lr_rms, hv_rms]

def fft_deviation(arr):
    fft_value = np.abs(np.fft.fft(arr))
    
    return fft_value.std()
    
def entropy2(labels):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    
    if n_labels <= 1:
        return 0
    
    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    
    if n_classes <= 1:
        return 0
    
    ent = 0.
    
    # Compute standard entropy.
    for i in probs:
        ent -= i * np.log(i, base=n_classes)
    
    return ent    
    
def getRubingInfo(GYRO_Z, look_a=4, delta=1500):
    """
    Out: Integer(total number of peaks), Float(average distance of plus peak)
    
    Retrun total peak occurences and average time among peaks 
    use GYRO_Z signal which is stable than EOG signal and precise enough to detect nose rubbing input
    """
    res = peakdetect.peakdetect(y_axis=GYRO_Z, lookahead=look_a, delta=delta)
    num_plus = len(res[0])
    num_minus = len(res[1])
    
    plus_xs = np.array(res[0])[:,0]
    mean_dx = np.average(np.diff(plus_xs))
    
    return num_plus+num_minus, mean_dx

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
    return plus/minus number count
    """
    res = peakdetect.peakdetect(y_axis=sig, lookahead=look_a, delta=delta)
    num_plus = len(res[0])
    num_minus = len(res[1])
    
    return num_plus, num_minus
    
    
    
def classifyLeftRight(EOG_H, EOG_V, printing=False):
    """
    In = EOG_H, EOG_V 
    Out = Integer
        0:neutral(cannot specify)
        1:left
        2:right
        
    Return the direction or rubbing status of Nose input 
    using EOG_H and EOG_V
    """
    sideH, H_plus, H_minus = sidebyValue(EOG_H)
    
#    print("H:(%d %f,%f)"%(sideH, H_plus, H_minus))
    if sideH == -999:
        if printing:
            print("classifyLeftRight| H: peack detection activated(%f,%f)"%(H_plus, H_minus))
        sideH = sidebyPeak(EOG_H)
    if sideH == 3:
        return 3    
        
    sideV, V_plus, V_minus = sidebyValue(EOG_V)
#    print("V:(%d %f,%f)"%(sideV, V_plus, V_minus))
    if sideV == -999:
        if printing:
            print("classifyLeftRight| V: peack detection activated(%f,%f)"%(H_plus, H_minus))
        sideV = sidebyPeak(EOG_V)
        
    if printing:
        print("classifyLeftRight| Side H,V: %d|%d"%(sideH,sideV))
    if sideH<0 and sideV<=0:
        return 1
    elif sideH>0 and sideV<=0:
        return 2
    else:
        return 0
"""
For data training using template data
"""
def getRMSs(cur_row):
#    data_dict = translateDFrowtoDict(cur_row)
    data_dict = cur_row
    
    lr_rms = 0
#    for val in ['mAvergeEOG_L', 'mAvergeEOG_R']:
    for val in ['EOG_L', 'EOG_R']:
        data = data_dict[val]
        if type(data)==np.ndarray:
            lr_rms += np.sqrt(np.square(data).mean())
        elif type(data)==pd.core.series.Series:
            lr_rms += np.sqrt(np.square(data.as_matrix()[0]).mean())
    hv_rms = 0
#    for val in ['mAvergeEOG_H', 'mAvergeEOG_V']:
    for val in ['EOG_H', 'EOG_V']:
        data = data_dict[val]
        if type(data)==np.ndarray:
            hv_rms += np.sqrt(np.square(data).mean())
        elif type(data)==pd.core.series.Series:
            hv_rms += np.sqrt(np.square(data.as_matrix()[0]).mean())
#    
    return lr_rms+hv_rms

def getAvgDiff(cur_row):
#    data_dict = translateDFrowtoDict(cur_row)
    data_dict = cur_row
    
    lr_rms = 0
#    for val in ['mAvergeEOG_L', 'mAvergeEOG_R']:
    for val in ['EOG_L', 'EOG_R']:
        data = data_dict[val]
        
        if type(data)==np.ndarray:
            lr_rms += np.abs(np.diff(data)).mean()
        elif type(data)==pd.core.series.Series:
            lr_rms += np.abs(np.diff(data.as_matrix())).mean()
#            lr_rms += np.diff(data.as_matrix()[0]).mean()
    hv_rms = 0
#    for val in ['mAvergeEOG_H', 'mAvergeEOG_V']:
    for val in ['EOG_H', 'EOG_V']:
        data = data_dict[val]
        if type(data)==np.ndarray:
            hv_rms += np.abs(np.diff(data)).mean()
        elif type(data)==pd.core.series.Series:
            hv_rms += np.abs(np.diff(data.as_matrix())).mean()
#            hv_rms += np.diff(data.as_matrix()[0]).mean()
#    
    return lr_rms+hv_rms

def pickTargetFromAllWindow(all_df, 
                            WINDOWS_SIZE = 100,
                            TARGET_VAL = ['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V', 
                                              "GYRO_X", "GYRO_Y", "GYRO_Z",
                                              "ACC_X", "ACC_Y", "ACC_Z"]):
    
    all_df_filtered = all_df.copy()
    all_df_filtered['TargetIndex'] = 0
    for index, row in all_df_filtered.iterrows():
        this_length = len(row[TARGET_VAL[0]])
        
        if this_length>WINDOWS_SIZE:
            cri_val = np.zeros(this_length)
            
            for end_index in range(WINDOWS_SIZE,this_length):
                start_index = end_index-WINDOWS_SIZE
                
                data_dict = dict()
                for one_col in TARGET_VAL:
                    data_dict[one_col] = row[one_col][start_index:end_index]
                cri_val[end_index] = getAvgDiff(data_dict)
            
            max_index = np.where(cri_val == cri_val.max())[0][0]
            
            for one_col in TARGET_VAL:
                all_df_filtered.at[index, one_col] = row[one_col][max_index-WINDOWS_SIZE:max_index]
            all_df_filtered.at[index,'TargetIndex'] = max_index
        else:
            all_df_filtered.at[index,'TargetIndex'] = -1*this_length
            
            
    return all_df_filtered
	
def makeTrainedRDFTree(f_name, folder="", output_folder="TrainedTree", save_fig=False):
    """
    Calculate all RMS for every windows (for calculation optimize within 10 frames)
    
    choose biggest EOG RMS as representative candidate
    -->make code to easily replace RMS functions to other function
    
    ADD X at exp_df and save again
    
    train RDF and save    
    -->Use gesture name on file name to easy to distinguish
    
    
    """
    exp_df =  pd.read_pickle(folder+f_name)
    
    exp_df['Candidate'] = ""
    exp_df['X'] = ""
    succeed_df = exp_df.loc[exp_df.TrialNum > -1]
    if save_fig:
        fig_folder = f_name.split(".")[0]
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)
            
    for i in succeed_df.index.values:
        data = succeed_df.loc[i].DATA.copy()
        
        feature_vals = np.zeros(len(data))
        for j in range(100, len(data)):
            cur_row = data.iloc[j-100:j]
#            feature_val = getRMSs(cur_row)
            feature_val = getAvgDiff(cur_row)
            
            feature_vals[j] = (feature_val)
            
        max_index = np.where(feature_vals == feature_vals.max())[0][0]
    
        cur_row = data.iloc[max_index-100:max_index]
        exp_df.at[i, 'Candidate'] = cur_row
        exp_df.at[i, 'X'] = makeX_wDict(cur_row)
        if save_fig:
            name = fig_folder+"/"+succeed_df.loc[i]['TargetName']+"_"+succeed_df.loc[i]['Name']+"_"+str(succeed_df.loc[i]['TrialNum'])
            plotResult(data, max_index-100, 100, name, save_fig)
            print(name,": (MAX:{})".format(max_index,feature_vals[max_index]))
            
    X = list(exp_df.loc[exp_df.TrialNum > -1].X)
    y = list(exp_df.loc[exp_df.TrialNum > -1].Target)
    
    useThis_Type_clf = RandomForestClassifier(n_estimators = 100)
    useThis_Type_clf.fit(X,y)
    scores = cross_val_score(RandomForestClassifier(n_estimators = 100), X, y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    gestures_list = [exp_df.loc[exp_df.Target == i].iloc[0]['TargetName'] for i in set(exp_df[exp_df.TrialNum > -1].Target)]
    gestures_str = ','.join(gestures_list)
    
    
    
    joblib.dump(useThis_Type_clf, output_folder+'/%s[%s]_Tree.pkl'%(f_name.replace("_wJINS.pickle",""),gestures_str)) 

    return exp_df

def plotResult(data, start_i=0, size=0, name="", save_fig=False):
    plt.figure()
    
	
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    
    f.suptitle(name)    
    ax1.set_title('EOG')
    ax2.set_title('GYRO')
    ax3.set_title('ACC')
    
    if not size==0:
        ax1.axvspan(start_i, start_i+size, color='red', alpha=0.3)
        ax2.axvspan(start_i, start_i+size, color='red', alpha=0.3)
        ax3.axvspan(start_i, start_i+size, color='red', alpha=0.3)
    
    for one_label in ["EOG_L", "EOG_R", "EOG_H", "EOG_V"]:
        try:
            ax1.plot(data[one_label].as_matrix(), label=one_label)
        except:
            ax1.plot(data[one_label], label=one_label)
    for one_label in ["GYRO_X", "GYRO_Y", "GYRO_Z"]:
        try:
            ax2.plot(data[one_label].as_matrix(), label=one_label)
        except:
            ax2.plot(data[one_label], label=one_label)
    for one_label in ["ACC_X", "ACC_Y", "ACC_Z"]:
        try:
            ax3.plot(data[one_label].as_matrix(), label=one_label)
        except:
            ax3.plot(data[one_label], label=one_label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    
    if save_fig:
        plt.savefig(name, dpi=300)
    else:
        plt.show()
        

def classifyBouncePush(EOG_L, EOG_R):
    """
    In = EOG_L, EOG_R 
    Out = Integer(side), Integer(type), Integer(number of exceeded signal)
        0: cannot specify
        1: bounce
        2: push
        
    Return direction,type of input and size of peak occured by the input
     1) claculate primary signal from (EOG_L, EOG_R) with maximum value
     2) if more than 22 value exceed threshold in primary signal , directly classify as push
     3) calculate side of each signal
     4) classify bounce/push by the side of primary and seconday signal
    """
    
    max_L = np.amax(EOG_L)
    max_R = np.amax(EOG_R)
    side = 0
    num_exceed = 0
    
    if max_L > max_R:
        side = 2
        num_exceed=maxNumConseExceed(EOG_L)
        if num_exceed>22:
            bounce_push = 2
            return side, bounce_push, num_exceed
        sideBig, L_plus, L_minus = sidebyValue(EOG_L)
        
        if sideBig == -999:
#            print("L: peack detection activated(%f,%f)"%(L_plus, L_minus))
            sideBig = sidebyPeak(EOG_L)
        sideSmall = sidebyPeak(EOG_R, delta=250)
        
    elif max_L < max_R:
        side = 1
        num_exceed=maxNumConseExceed(EOG_R)
        if num_exceed>22:
            bounce_push = 2
            return side, bounce_push, num_exceed
        sideBig, R_plus, R_minus = sidebyValue(EOG_R)
        
        if sideBig == -999:
#            print("R: peack detection activated(%f,%f)"%(R_plus, R_minus))
            sideBig = sidebyPeak(EOG_R)
        sideSmall = sidebyPeak(EOG_L, delta=250)
    else:
        sideBig = 0
    
    if sideBig == 0:
        # cannot define
        bounce_push=0
        return side, bounce_push, num_exceed
        
    if sideBig * sideSmall >0:
        # bounce
        bounce_push = 1
    else:
        # push
        if num_exceed < 12:
            bounce_push = 1
        bounce_push = 2
    
    
#    print("sideBig,sideSmall: %d,%d"%(sideBig,sideSmall))
    return side, bounce_push, num_exceed
    
    
    
    
    
    
    
class DispersionThre:
    def __init__(self, lag=10, thre=300, influence=0.5):
        self.lag = lag
        self.thre = thre
        self.influence = influence
        
        self.initialized = False
#==============================================================================
#         self.cur_signal = np.zeros(self.w_size)
#         self.avgFilter = np.zeros(self.w_size)
#         self.stdFilter = np.zeros(self.w_size)
#         self.filteredSig = np.zeros(self.w_size)
#         self.out_sig = np.zeros(self.w_size)
#==============================================================================
        
    def initialize(self,new_signal, continuing=False):
        if continuing:
            self.w_size = len(new_signal)
            
            self.cur_signal = np.append(self.cur_signal[-self.lag:],new_signal)
            self.avgFilter = np.append(self.avgFilter[-self.lag:],np.zeros(self.w_size))
            self.stdFilter = np.append(self.stdFilter[-self.lag:],np.zeros(self.w_size))
            self.filteredSig = np.append(self.filteredSig[-self.lag:],np.zeros(self.w_size))
            self.out_sig = np.append(self.out_sig[-self.lag:],np.zeros(self.w_size))
            
            self.w_size += self.lag
                
        else:
            self.w_size = len(new_signal)
            
            self.cur_signal = new_signal
            self.avgFilter = np.zeros(self.w_size)
            self.stdFilter = np.zeros(self.w_size)
            self.filteredSig = np.zeros(self.w_size)
            self.out_sig = np.zeros(self.w_size)
            
            
            
            self.avgFilter[self.lag-1] = np.mean(self.cur_signal[:self.lag]) 
            self.stdFilter[self.lag-1] = np.std(self.cur_signal[:self.lag]) 
            
            self.filteredSig[:self.lag] = self.cur_signal[:self.lag]
            
        
    def getResult(self, new_signal, continuing=False):
        self.initialize(new_signal, continuing=False)
            
        for i in range(self.lag, self.w_size):
            if np.abs(self.cur_signal[i] - self.avgFilter[self.lag-1])> self.thre*self.stdFilter[self.lag-1]:
                if self.cur_signal[i] > self.avgFilter[self.lag-1]:
                    self.out_sig[i] = 1
                else:
                    self.out_sig[i] = -1
                self.filteredSig[i] = self.influence*self.out_sig[i] + (1-self.influence)*self.filteredSig[self.lag-1]
                self.avgFilter[i] = np.mean(self.filteredSig[i-self.lag:i+1])
                self.avgFilter[i] = np.std(self.filteredSig[i-self.lag:i+1])
            else:
                self.out_sig[i] = 0
                self.filteredSig[i] = self.cur_signal[i]
                self.avgFilter[i] = np.mean(self.filteredSig[i-self.lag:i+1])
                self.avgFilter[i] = np.std(self.filteredSig[i-self.lag:i+1])
        
        if continuing:
            indexes = range(self.lag)
            self.cur_signal = np.delete(self.cur_signal,indexes)
            self.avgFilter = np.delete(self.avgFilter,indexes)
            self.stdFilter = np.delete(self.stdFilter,indexes)
            self.filteredSig = np.delete(self.filteredSig,indexes)
            self.out_sig = np.delete(self.out_sig,indexes)
                
        return self.out_sig
        
        


    
    
    
    
    
    