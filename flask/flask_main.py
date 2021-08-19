# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:50:38 2020

@author: kctm
"""

from flask import Flask
from flask import request, render_template, jsonify, redirect, url_for

# from flaskwebgui import FlaskUI #get the FlaskUI class

import pygame

import threading, webbrowser

import time
from datetime import datetime

import os, sys

import numpy as np
import pandas as pd
from scipy import signal

import pickle
from joblib import dump, load


import inspect


import random
import json
from flask import Response


# import custom
sys.path.append("../libs")
from sensorUDP import imus_UDP
import JinsSocket
from NoseExperiment_clean import Experiment 
from pygameDisplay import showResult  
import methods_filter, methods_feature, methods_model



pygame_is_running = False
############ SETTINGS [start] ############
# set Experiment mode
participant_name = "P0" # put name of participant
number_of_trials = 5
#target_gestures = ["Nothing","Left Flick", "Left Push", "Right Flick", "Right Push", "Rubbing"]
# target_gestures = ["Face touch", "null"]
target_gestures = ['nose right', 'nose left',
                   'left eye', 'right eye',
                   'mouth left', 'mouth right',
                   'null']


enable_experiment = True # set False for just testing classifier
isTraining = False
save_result = True  #set as True to save automatically
save_folder = "CollectedData"
save_trained_folder = "TrainedModel"
save_plot_figure = True
experiment_mode = 1 #1: auto time count, 2:wait till succeed

time_before = 2     #for all experiment mode. sec before start recording after press key
time_recording = 2 # only for experiment mode 1

showFPS = False


# Set saving size of Jins MEME
# @@@@@@saving size matters the speed and framerate of calculation
isUDP = True
Port = 12562
saving_size = 100*60*10


# RDF parameters
sampling_window = 1000
stab_time = 500

enable_freq_showing = False
enable_graph_showing = False
print_status = False
############ SETTINGS [end] ############



app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True





@app.route('/_initdata', methods= ['GET'])
def info_to_html():
    print("<<stuff>>\nname:",participant_name,
          "\ntarget_gestures:",target_gestures,
          "\nnumber_of_trials:",number_of_trials)
    return jsonify(participant_name=participant_name,
                   trial_numbers=number_of_trials,
                   target_gestures=",".join(target_gestures),
                   time_before=time_before,
                   time_recording=time_recording)

@app.route('/eog-data')
def jins_data():
    def get_jins_data():
        global jins_client
        last_t = 0
        while True:
            new_results = jins_client.getLast_dict_one()
            
            if new_results['TIME'] != last_t:
                json_data = json.dumps(new_results)
                
                last_t = new_results['TIME']
                yield f"data:{json_data}\n\n"
            time.sleep(0.001)

    return Response(get_jins_data(), mimetype='text/event-stream')

@app.route('/imu-data')
def imu_data():
    def get_imu_data():
        global imu_get
        last_t = 0
        while True:
            new_results = imu_get.getLastData()
            
            
            if type(new_results)!=int and new_results[-5] != last_t:
                # print( new_results[-3:])
                json_data = json.dumps({'TIME': new_results[-5],
                                        'MAG_X': new_results[-3],
                                        'MAG_Y': new_results[-2],
                                        'MAG_Z': new_results[-1]
                    })
                
                last_t = new_results[-5]
                yield f"data:{json_data}\n\n"
            time.sleep(0.001)

    return Response(get_imu_data(), mimetype='text/event-stream')


@app.route('/_gestureplot', methods= ['GET','POST'])
def send_gesture_plot_info():
    global selected_seg_data
    print("===startReview")
    if request.method == 'POST':
        selected_seg_data = request.form['selected_seg_data']
                
    
    segmented_df = pd.read_pickle(os.path.join(save_folder, selected_seg_data))
    all_trials_list = segdf_to_chartdict(segmented_df)
    return jsonify(all_trials=all_trials_list)


result_html = "Press 'Start Training' button to get result"
@app.route('/_traintest', methods= ['GET','POST'])
def train_test_gestures():
    global save_folder, result_html
        
    if request.method == 'POST':
        selected_train_data = pd.read_pickle(os.path.join(save_folder,request.form['selected_train_data']))
        
        TARGET_FILTER = split_remove_empty(request.form['checked_filter'])
        TARGET_MODEL = request.form['checked_model']
        TARGET_RAW_AXIS = split_remove_empty(request.form['checked_raw_data_input']) 
        TARGET_FEATURE_AXIS = split_remove_empty(request.form['checked_target_axis'])
        TARGET_FEATURES = split_remove_empty(request.form['checked_features']) 
        result_html = "training.... please wait"
        
        totalX, totaly, target_names_list = create_Xy_from_df(selected_train_data, TARGET_FILTER, TARGET_RAW_AXIS,TARGET_FEATURE_AXIS,TARGET_FEATURES)
        # result_str = get_train_result(totalX, totaly, target_names_list, TARGET_MODEL)
        
        model = getattr(methods_model, TARGET_MODEL)()
        results_ = model.get_confusion_matrix(totalX,totaly,cv=2, target_names_list=target_names_list)
        conf_mat_html = methods_model.Classifier.confmat_to_htmltable(results_[1],model.target_names_list)
        result_html = "Last run:"+datetime.now().strftime('%Y-%m-%d %H_%M_%S')+"<br><br>"\
                        +"Model: {}".format(TARGET_MODEL)+"<br>"\
                        +"mean: {:.2f}%,  std: {:.2f}%".format(results_[0].mean()*100, results_[0].std()*100)+"<br><br>"\
                        +conf_mat_html+"<br><br><br><br>"
        
    return jsonify(result=result_html)
@app.route('/_refreshresult', methods= ['GET','POST'])
def resfresh_train_results():
    global result_str

    return jsonify(result=result_str)

model_saved_name = "notyet saved"
@app.route('/_savemodel', methods= ['GET','POST'])
def save_trained_model():
    global save_trained_folder, model_saved_name
        
    if request.method == 'POST':
        selected_train_data = pd.read_pickle(os.path.join(save_folder,request.form['selected_train_data']))
        
        TARGET_FILTER = split_remove_empty(request.form['checked_filter'])
        TARGET_MODEL = request.form['checked_model']
        TARGET_RAW_AXIS = split_remove_empty(request.form['checked_raw_data_input']) 
        TARGET_FEATURE_AXIS = split_remove_empty(request.form['checked_target_axis'])
        TARGET_FEATURES = split_remove_empty(request.form['checked_features']) 
        
        totalX, totaly, target_names_list = create_Xy_from_df(selected_train_data, TARGET_FILTER, TARGET_RAW_AXIS,TARGET_FEATURE_AXIS,TARGET_FEATURES)
        
        
        model = getattr(methods_model, TARGET_MODEL)()
        model.train(totalX,totaly,target_names_list=target_names_list)
        
        saving_dict = {"TARGET_FILTER" : TARGET_FILTER,
                       "TARGET_MODEL": TARGET_MODEL,
                       "TARGET_RAW_AXIS": TARGET_RAW_AXIS,
                       "TARGET_FEATURE_AXIS": TARGET_FEATURE_AXIS,
                       "TARGET_FEATURES": TARGET_FEATURES,
                       "model": model}
        
        f_name = os.path.join(save_trained_folder, datetime.now().strftime('%Y-%m-%d %H_%M_%S')+"_{}.joblib".format(model.__class__.__name__))
        dump(saving_dict,f_name)
        # model.save_model(f_name)
        model_saved_name = f_name
        print("----saved:",model_saved_name)
    return jsonify(result=model_saved_name)





# @app.route('/_refreshresult', methods= ['GET','POST'])
# def resfresh_train_results():
#     global result_str

#     return jsonify(result=result_str)

@app.route('/', methods=['GET', 'POST'])
def init_data_gathering():
    global participant_name, target_gestures, number_of_trials, pygame_is_running, enable_experiment, isTraining, save_result
    
    
    # for button
    if request.method == 'POST':
        participant_name = request.form['input_name'].upper()
       
        target_gestures = [x.strip() for x in request.form["input_gesture_set"].split(',')]
        
        number_of_trials = int(request.form['input_number_of_gesture'])
        
        time_before = float(request.form['time-before-length'])
        time_recording = float(request.form['time-window-length'])
        print("\n<<init_data_gathering>>\nname:",participant_name,
              "\ntarget_gestures:",target_gestures,
              "\nnumber_of_trials:",number_of_trials,
              "\ntime_before:",time_before,
              "\ntime_recording:",time_recording,
              "\n\n")
        
        # if not 'action' in request.form:
        #     return render_template('form.html', name=participant_name)
        
        
        if request.form['action'] == 'startGathering':
            if not pygame_is_running:
                runPygame(participant_name, number_of_trials, target_gestures,
                          enable_experiment=enable_experiment, save_result=save_result)
        elif request.form['action'] == 'switchTraining':
            return redirect(url_for('review_data'))

    return render_template('form.html', name=participant_name)

@app.route('/online_plot', methods=['GET', 'POST'])
def online_plot():
    global jins_client, imu_get
    if not 'jins_client' in globals():
        jins_client = JinsSocket.JinsSocket(isUDP=True, Port=12562, w_size=100*60*5)
        jins_client.setConnection()
        jins_client.start()
    if not 'imu_get' in globals():
        imu_get = imus_UDP(Port=12563)
        imu_get.setConnection()
        imu_get.start()
    return render_template('online_plot.html')
    
@app.route('/review_data', methods=['GET', 'POST'])
def review_data():
    global save_folder
    
    error = None
    exp_list = refreshtrainingDataList(save_folder)
    seg_list = refresh_segDataList(save_folder)
    
    if request.method == 'POST':
            
        if request.form['action'] == 'runSeg':
            
            selected_exp = request.form.get("target_data_selection")
            selected_segment_method = request.form.get("segment_type")
            
            print("===startReview")
            print("selected_exp: ",selected_exp)
            print("selected_segment_method: ",selected_segment_method)
            
            trial_schedule_df = pd.read_pickle(os.path.join(save_folder,selected_exp+".pickle"))
            jins_data_df = pd.read_csv(os.path.join(save_folder,selected_exp+"_JINS.csv"), index_col=0)
            jins_data_df = jins_data_df[(jins_data_df.T != 0).any()]
            if len(jins_data_df)<100:
                error = selected_exp + f"- no proper sensor data collected==(too small jins data:{len(jins_data_df)}frames)"
            elif 'Basic' in selected_segment_method :
                new_df = putIMUinDF(trial_schedule_df, jins_data_df)
                if len(new_df)==0:
                    error = selected_exp + "- no proper sensor data collected"
                else:
                    new_df.to_pickle(os.path.join(save_folder,selected_exp+"_segmented.pickle"))
                    seg_list = refresh_segDataList(save_folder)
            else:
                error = "Does not support this segment method: {}".format(selected_segment_method)
                
    return render_template('review_data.html',
                           available_exp=exp_list,
                           available_seg=seg_list,
                           error=error)

@app.route('/training', methods=['GET', 'POST'])
def training():
    global save_folder
    
    seg_list = refresh_segDataList(save_folder)
    
    filter_list = [a[0] for a in inspect.getmembers(methods_filter, inspect.isfunction)]
    models_list = [a[0] for a in inspect.getmembers(methods_model, inspect.isclass) if 'methods_model' in str(a[1])]
    models_list.remove('Classifier')
    features_list = [a[0] for a in inspect.getmembers(methods_feature, inspect.isfunction)]

    return render_template('training.html',
                           available_seg=seg_list,
                           available_filter = filter_list,
                           available_model = models_list,
                           available_features=features_list)


@app.route('/online_test', methods=['GET', 'POST'])
def online_test():
    global number_of_trials, target_gestures
    model_list = refresh_modelList(save_trained_folder)
    if request.method == 'POST':
        print("===start online test")
        f_name_model = os.path.join(save_trained_folder, request.form.get("target_model_selection"))
        print(f_name_model)
        
        if request.form['action'] == 'startOnlinetest':
            if not pygame_is_running:
                runPygame(participant_name="test", trial_numbers=number_of_trials, target_gestures=target_gestures,
                      one_dollar_template = None,
                      enable_experiment=False, save_result=False,
                      show_online = True,
                      model_name = f_name_model)
        
    return render_template('online_test.html', available_model=model_list)




#########################################
########### Functions [start] ###########
current_milli_time = lambda: int(round(time.time() * 1000))
current_milli_time_f = lambda: float(time.time() * 1000)
def split_remove_empty(str_, sep=','):
    list_ = str_.split(sep)
    
    if '' in list_:
        list_.remove('')
    return list_
def checkFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def refreshtrainingDataList(save_folder):
    all_pickles = [f for f in os.listdir(save_folder) if '.pickle' in f]
    exp_names = [f.split(".")[0] for f in all_pickles if 'EXP1.pickle' in f]
    

    exps_has_jins = [f for f in exp_names if os.path.exists(os.path.join(save_folder,f+"_JINS.csv"))]
    # for f in exp_names:
    #     jins_name = os.path.join(save_folder,f+"_JINS.csv")
    #     if os.path.exists(jins_name):
    #         jins_df = pd.read_csv(jins_name,index_col=0)
    #         jins_df = jins_df[(jins_df.T != 0).any()]
    #         if len(jins_df)>100:
    #             exps_has_jins.append(f)
    return reversed(exps_has_jins)
def refresh_segDataList(save_folder):
    all_pickles = [f for f in os.listdir(save_folder) if 'segmented.pickle' in f]
    return reversed(all_pickles)
def refresh_modelList(save_folder):
    all_pickles = [f for f in os.listdir(save_folder) if '.joblib' in f]
    return reversed(all_pickles)                        
def putIMUinDF(data_df, imu_df, post_fix=""):
    new_df = data_df.loc[data_df.Target>0].copy()
    for i,row in new_df.iterrows():
        start_t = row['StartTime']
        end_t = row['CurrentTime']
        
        this_trial_imu_df = imu_df.loc[(start_t<imu_df.EpochTime) & (imu_df.EpochTime<end_t)]
        if len(this_trial_imu_df)>0:
            new_df.at[i, 'DATA%s'%post_fix] = this_trial_imu_df
        else:
            new_df.drop([i], inplace=True)
        
    return new_df

def putTrialinIMU(data_df, imu_df, name=""):
    cols = [ a for a in list(data_df.columns) if not 'DATA' in a]
    new_imu_df = imu_df.copy()
    for col in cols:
        new_imu_df[col] = np.nan
    
    for i, this_row in data_df.iterrows():
        start_t = this_row['StartTime']
        end_t = this_row['CurrentTime']
        
        new_imu_df.at[new_imu_df.loc[(new_imu_df.EpochTime>start_t) & (new_imu_df.EpochTime<end_t)].index, cols] = list(this_row)
    
    return new_imu_df



def segdf_to_chartdict(segmented_df):
    all_charts_info = []
    for one_gesture in segmented_df.TargetName.unique():
        
    
        this_gesture_df = segmented_df.loc[segmented_df.TargetName==one_gesture]
        if one_gesture == 'EndPoint' and this_gesture_df.iloc[0].TrialNum<0:
            continue
        
        trials = []
        for i,one_row in this_gesture_df.iterrows():
            DATA = one_row.DATA
            
            sensor_vals = dict()
            for one_sensor in ['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'ACC_X', 'ACC_Y', 'ACC_Z']:
            # for one_sensor in ['EOG_H', 'EOG_V']:
                sensor_vals[one_sensor] = list(DATA[one_sensor])
                
            row_dict = {'tick_label': list(DATA['EpochTime']),
                        'sensor_vals': sensor_vals,
                        'id': one_row.TrialNum}
            trials.append(row_dict)
            
        one_chart_info = {'name': one_gesture,
                          'trials': trials}
        
        all_charts_info.append(one_chart_info)
        
    return all_charts_info 


    
def makeoneDollarTrainingSet(f_name):
    data_df = pd.read_pickle(f_name)
    custom_training = dict()
    
    only_success = data_df.loc[data_df.TrialNum >= 0]
    for i in range(max(only_success.Target)+1):
        one_target_trials = only_success.loc[only_success.Target==i]
        
        try:
            this_name = one_target_trials.iloc[0].TargetName
        except:
            this_name = str(one_target_trials.iloc[0].Target)
        
        this_data_list = []
        for k in range(len(one_target_trials)):
            this_data_df = one_target_trials.iloc[k].DATA.loc[one_target_trials.iloc[k].DATA.Ring_input==1]
            this_data_arr = np.array(this_data_df[['Diff_x', 'Diff_y']])
#            this_data_list.append([convertDifftoXY(this_data_arr[j,0], this_data_arr[j,1]) for j in range(len(this_data_arr))])
            this_data_list.append([(this_data_arr[j,0], this_data_arr[j,1]) for j in range(len(this_data_arr))])
        custom_training[this_name] = this_data_list
        
    with open(f_name.replace("_wIMU.pickle", "_onedollarTrain.pickle"), 'wb') as handle:
        pickle.dump(custom_training, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        print("DONE: %s"%f_name)
def create_Xy_from_df(target_df, TARGET_FILTER, TARGET_RAW_AXIS,TARGET_FEATURE_AXIS,TARGET_FEATURES):
    this_features_sum = methods_feature.sumAllFeatures(TARGET_FEATURES)
    
    totalX = []
    totaly = []
    
    AXIS_SET = {'EOG': ['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V'],
                'ACC':  ['ACC_X', 'ACC_Y', 'ACC_Z'],
                'GYRO': ['GYRO_X', 'GYRO_Y', 'GYRO_Z']}
    W_LENGTH = 50
    
    target_names_list = [target_df.loc[target_df.Target==i].iloc[0].TargetName for i in target_df.Target.unique() if i>=0]
    
    # create total label
    row = target_df.iloc[0]
    total_label = []
    this_data = row.DATA.copy()
    for one_axis_set in TARGET_RAW_AXIS:
        if '_fft' in one_axis_set:
            target_set = one_axis_set.replace('_fft','')
            is_fft = True
        else:
            target_set = one_axis_set
            is_fft = False
        
        for one_part in AXIS_SET[target_set]:
                if is_fft:
                    post_fix = 'fft'
                else:
                    post_fix = ''
                total_label += [one_part+"{}_{:03d}".format(post_fix,i) for i in range(W_LENGTH)]
    for one_axis in TARGET_FEATURE_AXIS:
        for one_part in AXIS_SET[target_set]:
            total_label += [one_axis+"_"+one_name for one_name in this_features_sum.update_label()]
              
                
    for i,row in target_df.iterrows():
        if row.TrialNum <0:
            continue
        this_data = row.DATA.copy()
        
        rowX = get_single_X(this_data, this_features_sum, TARGET_FILTER, TARGET_RAW_AXIS,TARGET_FEATURE_AXIS)
                
        totalX.append(list(rowX))
        totaly.append(row.Target)
        
    return totalX, totaly, target_names_list

def get_single_X(this_data, sum_feature, TARGET_FILTER, TARGET_RAW_AXIS,TARGET_FEATURE_AXIS,
                             AXIS_SET = {'EOG': ['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V'],
                                            'ACC':  ['ACC_X', 'ACC_Y', 'ACC_Z'],
                                            'GYRO': ['GYRO_X', 'GYRO_Y', 'GYRO_Z']},
                            fft_resample_num = 50):
    
    rowX = np.array([])
    for one_axis_set in TARGET_RAW_AXIS:
            if '_fft' in one_axis_set:
                target_set = one_axis_set.replace('_fft','')
                is_fft = True
            else:
                target_set = one_axis_set
                is_fft = False
            
            for one_part in AXIS_SET[target_set]:
                try:
                    tmp_raw = signal.resample(this_data[one_part].values, fft_resample_num)
                except:
                    tmp_raw = signal.resample(this_data[one_part], fft_resample_num)
                        
                if is_fft:
                    tmp_raw = np.abs(np.fft.fft(tmp_raw))
                rowX = np.append(rowX, tmp_raw)
                    
            
    for one_axis in TARGET_FEATURE_AXIS:
        if 'diff' in one_axis:
            target_set = one_axis.replace('_diff','')
            is_diff = True
        else:
            target_set = one_axis
            is_diff = False
            
        for one_part in AXIS_SET[target_set]:
            try:
                sig = this_data[one_part].values
            except:
                sig = this_data[one_part]
                
            if is_diff:
                sig = np.diff(sig)
            for one_filter in TARGET_FILTER:
                sig = getattr(methods_filter, one_filter)(sig)
            partX = sum_feature.cal_features_from_one_signal(sig)
            
            rowX = np.append(rowX, partX)
                
    return rowX
    
def get_train_result(totalX, totaly, target_names_list, TARGET_MODEL):
    
    model = getattr(methods_model, TARGET_MODEL)()
    results_ = model.get_confusion_matrix(totalX,totaly,cv=2,target_names_list=target_names_list)
    
    result_str = "Last run:"+datetime.now().strftime('%Y-%m-%d %H_%M_%S')+"\nModel: {}".format(TARGET_MODEL)+"\n"\
                    +"mean: {:.2f}%,  std: {:.2f}%".format(results_[0].mean()*100, results_[0].std()*100)+"\n"\
                    +str(target_names_list)+"\n"\
                    +str(results_[1])+"\n\n\n"
    # result_str += conf_matrix_fixedwidth(results_[1], target_names_list)
    print(result_str)
    return result_str

def conf_matrix_fixedwidth(conf_mat, target_list):
    str_ = ""
    num = len(target_list)
    max_str_len = max([len(one) for one in target_list])
    
    target_list_str = [one.rjust(max_str_len," ") for one in target_list]
    
    str_ += (" "*(max_str_len+2)) +"|".join(target_list_str)+"\n"
    for i in range(num):
        one_row_str = [str(one).rjust(max_str_len," ") for one in conf_mat[i,:]]
        str_ += target_list_str[i]+" |"+",".join(one_row_str)+"\n"
    return str_
"""
FUNCTIONS FOR KEY/MOUSE EVENT in PyGame

"""

        
def findLatest(jins_class, orbit_class):
    t1 = jins_class.TIME[-1]
    t2 = orbit_class.TIME[-1]
    
    if t1>t2:
        return t2
    else:
        return t1
        
def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
           


    
#########################################
########### Functions [end] ###########

    
def runPygame(participant_name, trial_numbers, target_gestures,
              one_dollar_template = None,
              model_name = "TrainedModel/2020-05-06 16_41_12_RDFclassifier.joblib",
              enable_experiment = True, save_result = False, show_online = False,
              width=1920, height=1080, full_screen = False,
              background = (200,200,200, 255),
              dt = 10):
    global pygame_is_running, jins_client
    pygame_is_running = True
    targetType = dict()
    for i, name in enumerate(target_gestures):
        targetType[i] = name
        
    # Loop until the user clicks the close button.
    done = False                 
         
    
    pygame.init()
    pygame.mixer.init()
    
    
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
    fps_font = pygame.font.SysFont('hack', 15)
    
    # Set the height and width of the screen
    size = [width, height]
    if full_screen:
        screen = pygame.display.set_mode(size,pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(size)
    
    # For opacity settings on background
    s = pygame.Surface((width, height), pygame.SRCALPHA)   # per-pixel alpha
    s.fill(background)   
    screen.blit(s, (0,0)) 
    
    if showFPS:
        fpsText = fps_font.render("%dfps"%(clock.get_fps()), True, (100,255,100))
        screen.blit(fpsText, (0,0))
    
    pygame.display.set_caption("Ocular Gesture Toolkit")
    
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
    
    
    
    """PyGame part"""
    # -------- Main Program Loop -----------
    

    tick_value = 1000/dt
    
    save_name_str = save_folder +"/"+ datetime.now().strftime('%Y-%m-%d %H_%M_%S')+"_"+participant_name+"_EXP%d"%(experiment_mode)
    
    """Thread 1: DATA COLLECTION """
    if not 'jins_client' in globals():
        jins_client = JinsSocket.JinsSocket(isUDP=True, Port=12562, w_size=saving_size, save_name=save_name_str)
        # jins_client = JinsSocket.JinsSocket(isUDP=False, Port=12562, w_size=saving_size, save_name=save_name_str)
        jins_client.setConnection()
        jins_client.start()
    

    exp1 = Experiment(experiment_mode,
                      name = participant_name, trial_num = trial_numbers, size = size,
                      typeText = targetType, tts=True) 
    #                  ,typeSound = typeSound)
    exp1.setDataCollection(time_before=time_before,time_recording=time_recording)
    
    if show_online:
        """load/init classifier"""
        load_model_and_info = load(model_name)
        
        feature_creator = methods_feature.sumAllFeatures(load_model_and_info['TARGET_FEATURES'])
        clf_model = load_model_and_info['model']
        
        type_forText = dict()
        for i, one_gesture in enumerate(clf_model.target_names_list):
            type_forText[i] = one_gesture
        show_pygame = showResult(pygame, screen, type_forText)
                 
        

    
    pygame_is_running = True
    while not done:
        
        """Pre-run"""
        screen.fill(background)
        
        
        
            
        """Keypress event"""
        # --- Event Processing
        events =  pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                done = True
                
            # User pressed down on a key
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
            # User let up on a key
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    exp1.rollbackOneTrial()
                elif event.key == pygame.K_RIGHT:
                    if experiment_mode in [1,2,3]:
                        exp1.startRecording()
                elif event.key == pygame.K_f:
                    if experiment_mode in [3]:
                        exp1.recording = False
                elif event.key == pygame.K_p:
                    if experiment_mode == 2:
                        print("have to make Passing function")
                        
        cur_t = current_milli_time()
        cur_res = -1
#        cur_res, cur_prop = rdf_class.runJinsTypeonly(cur_t, printing=print_status)
        
                
        
        if enable_experiment:
            """for data collection"""
            
            if experiment_mode in [1,2,3]:
                target_res = exp1.countDetectionResult(cur_res)
                current_state = exp1.current_state
                toggle = exp1.checkStateChange() 
                exp1.drawPyGame1(screen)
                
                
        elif show_online:
            """for online test"""
            window_data = jins_client.getLastbyTime_dict(2000)
            inputX = get_single_X(window_data, feature_creator,
                                  load_model_and_info['TARGET_FILTER'],
                                  load_model_and_info['TARGET_RAW_AXIS'],
                                  load_model_and_info['TARGET_FEATURE_AXIS'])
            
            inputX = inputX.reshape(1, -1)
            
            cur_res, cur_prop = clf_model.classify_w_prob(inputX)
            # cur_res, cur_prop = rdf_class.runJinsTypeonly(cur_t, printing=print_status)
            
            if len(cur_prop)>0:
                show_pygame.showResultTextwProp(cur_res, cur_prop)
        else:
            print("need testing UI")
        
            
        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
        # --- Wrap-up
        # Limit to 60 frames per second
        clock.tick(tick_value)
        
    # Close everything down
    jins_client.close()
    pygame.quit()
    pygame_is_running = False
    # jins_client.close()
    
    
    if save_result and enable_experiment and len(exp1.trialDF)>0:
        
        try:
            
        
            exp1.trialDF.to_csv(save_name_str+".csv")
            exp1.trialDF.to_pickle(save_name_str+".pickle")
        
            # jins_df = jins_client.saveasPickle(save_name_str+"_EXP%d"%(experiment_mode))
            # new_jinsDF = JinsSocket.addMovingAverage(jins_df)
            # new_jinsDF.to_pickle('%s_JINS.pickle'%(save_name_str+"_EXP%d"%(experiment_mode)))
        
        
            # new_df = putIMUinDF(exp1.trialDF, new_jinsDF)
            # new_df.to_pickle(save_name_str+"_EXP%d_wJINS.pickle"%(experiment_mode))
            # new_df.to_csv(save_name_str+"_EXP%d_wJINS.csv"%(experiment_mode))
            
#            new_imu_df = putTrialinIMU(exp1.trialDF, new_jinsDF)
#            new_imu_df.to_csv(save_name_str+"_EXP%d.csv"%(experiment_mode))
#            new_imu_df.to_pickle(save_name_str+"_EXP%d.pickle"%(experiment_mode))
            print("Result saved")
        except:
            print("Error on combining EXP/JINS data")
            
            
# call the 'run' method
if __name__ == '__main__':
    
    

    checkFolder(save_folder)
    checkFolder(save_trained_folder)


    url = "http://127.0.0.1:5000"
    # threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    
    # Feed it the flask app instance 
    # ui = FlaskUI(app)
    # ui.run()
    
    app.run()
    
    
    