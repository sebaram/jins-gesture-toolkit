# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:50:38 2020

@author: kctm
"""

from flask import Flask
from flask import request, render_template, jsonify, redirect, url_for

from flaskwebgui import FlaskUI #get the FlaskUI class

import pygame

import threading, webbrowser

import time
from datetime import datetime

import os, sys

import numpy as np
import pandas as pd
import pickle


# import custom
sys.path.append("../libs")
import JinsSocket
import NoseTools as NoseTools
from NoseExperiment_clean import Experiment 
from pygameDisplay import showResult  



pygame_is_running = False
############ SETTINGS [start] ############
# set Experiment mode
participant_name = "P0" # put name of participant
number_of_trials = 5
#target_gestures = ["Nothing","Left Flick", "Left Push", "Right Flick", "Right Push", "Rubbing"]
target_gestures = ["Eye Movement", "Blink"]


enable_experiment = True # set False for just testing classifier
isTraining = False
save_result = True  #set as True to save automatically
save_folder = "CollectedData"
save_trained_folder = "TrainedModel"
save_plot_figure = True
experiment_mode = 1 #1: auto time count, 2:wait till succeed

time_before = 2     #for all experiment mode. sec before start recording after press key
time_recording = 3 # only for experiment mode 1

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
    return jsonify(participant_name=participant_name, trial_numbers=number_of_trials, target_gestures=",".join(target_gestures))


@app.route('/', methods=['GET', 'POST'])
def init_data_gathering():
    global participant_name, target_gestures, number_of_trials, pygame_is_running, enable_experiment, isTraining, save_result
    
    
    # for button
    if request.method == 'POST':
        participant_name = request.form['input_name'].upper()
       
        target_gestures = [x.strip() for x in request.form["input_gesture_set"].split(',')]
        
        number_of_trials = int(request.form['input_number_of_gesture'])
        print("<<init_data_gathering>>\nname:",participant_name,
              "\ntarget_gestures:",target_gestures,
              "\nnumber_of_trials:",number_of_trials)
        
        if request.form['action'] == 'startGathering':
            if not pygame_is_running:
                runPygame(participant_name, number_of_trials, target_gestures,
                          enable_experiment=enable_experiment, save_result=save_result)
        elif request.form['action'] == 'switchTraining':
            return redirect(url_for('review_data'))

    return render_template('form.html', name=participant_name)

@app.route('/review_data', methods=['GET', 'POST'])
def review_data():
    global save_folder
    
    exp_list = refreshtrainingDataList(save_folder)
    
    if request.method == 'POST':
        if request.form['action'] == 'startReview':
            print("startReview")
    return render_template('review_data.html', available_exp=exp_list)

@app.route('/online_test', methods=['GET', 'POST'])
def online_test():

    return render_template('online_test.html')

@app.route('/training', methods=['GET', 'POST'])
def training():

    return render_template('training.html')


#########################################
########### Functions [start] ###########
current_milli_time = lambda: int(round(time.time() * 1000))
current_milli_time_f = lambda: float(time.time() * 1000)

def checkFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def refreshtrainingDataList(save_folder):
    all_pickles = [f for f in os.listdir(save_folder) if '.pickle' in f]
    exp_names = [f.split(".")[0] for f in all_pickles if 'EXP1.pickle' in f]
    
    exps_has_jins = [f for f in exp_names if os.path.exists(os.path.join(save_folder,f+"_JINS.pickle"))]
    return reversed(exps_has_jins)
            
def putIMUinDF(data_df, imu_df, post_fix=""):
    new_df = data_df.copy()
    for i in range(len(new_df)):
        start_t = new_df.loc[i]['StartTime']
        end_t = new_df.loc[i]['CurrentTime']
        
        this_trial_imu_df = imu_df.loc[(imu_df.EpochTime>start_t) & (imu_df.EpochTime<end_t)]
        new_df.at[i, 'DATA%s'%post_fix] = this_trial_imu_df
        
    return new_df

def putTrialinIMU(data_df, imu_df, name=""):
    cols = [ a for a in list(data_df.columns) if not 'DATA' in a]
    new_imu_df = imu_df.copy()
    for col in cols:
        new_imu_df[col] = np.nan
    
    for i in range(len(data_df)):
        this_row = data_df.loc[i][cols]
        start_t = data_df.loc[i]['StartTime']
        end_t = data_df.loc[i]['CurrentTime']
        
        new_imu_df.at[new_imu_df.loc[(new_imu_df.EpochTime>start_t) & (new_imu_df.EpochTime<end_t)].index, cols] = list(this_row)
    
    return new_imu_df
    
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
              forest_TF_name = save_trained_folder+"/Default_TF_Tree.pkl", forest_Type_name = save_trained_folder+"/Default_Type_Tree.pkl",
              enable_experiment = True, save_result = False, show_online = False,
              width=1920, height=1080, full_screen = False,
              background = (200,200,200, 255),
              dt = 10):
    global pygame_is_running, myWindow
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
    
    pygame.display.set_caption("Jins MEME Gesture Toolkit")
    
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
    
    
    
    """PyGame part"""
    # -------- Main Program Loop -----------
    

    tick_value = 1000/dt
    
    """Thread 1: DATA COLLECTION """
    jins_client = JinsSocket.JinsSocket(isUDP=True, Port=12562, w_size=saving_size)
    jins_client.setConnection()
    jins_client.start()


    exp1 = Experiment(experiment_mode,
                      name = participant_name, trial_num = trial_numbers, size = size,
                      typeText = targetType, tts=True) 
    #                  ,typeSound = typeSound)
    exp1.setDataCollection(time_before=time_before,time_recording=time_recording)
    
    if show_online:
        if "Default_Type_Tree.pkl" in forest_Type_name:
            show_pygame = showResult(pygame, screen)
        else:
            type_forText = dict()
            gestures = forest_Type_name.split("[")[1].split("]")[0].split(",")
            
            for i, one_gesture in enumerate(gestures):
                type_forText[i] = one_gesture
            
            show_pygame = showResult(pygame, screen, type_forText)
        
        

    if show_online:
        rdf_class = NoseTools.RDFclassifier(target_thre_noFalse=0.60, sampling_dt=sampling_window, 
                                        stab_time=stab_time, dt_ms=10, 
                                        jins_client=jins_client, enable_fft_graph=enable_freq_showing,
                                        enable_graph=enable_graph_showing, 
                                        f_TF=forest_TF_name, f_Type=forest_Type_name)
    
    
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
            
            
            
            if experiment_mode in [1,2,3]:
                target_res = exp1.countDetectionResult(cur_res)
                current_state = exp1.current_state
                toggle = exp1.checkStateChange() 
                exp1.drawPyGame1(screen)
                
                
        elif show_online:
            cur_res, cur_prop = rdf_class.runJinsTypeonly(cur_t, printing=print_status)
            
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
    pygame.quit()
    pygame_is_running = False
    jins_client.close()
    
    
    if save_result and enable_experiment:
        
        try:
            save_name_str = save_folder +"/"+ datetime.now().strftime('%Y-%m-%d %H_%M_%S')+"_"+participant_name
        
            exp1.trialDF.to_csv(save_name_str+"_EXP%d.csv"%(experiment_mode))
            exp1.trialDF.to_pickle(save_name_str+"_EXP%d.pickle"%(experiment_mode))
        
            jins_df = jins_client.saveasPickle(save_name_str+"_EXP%d"%(experiment_mode))
            new_jinsDF = JinsSocket.addMovingAverage(jins_df)
            new_jinsDF.to_pickle('%s_JINS.pickle'%(save_name_str+"_EXP%d"%(experiment_mode)))
        
        
            # new_df = putIMUinDF(exp1.trialDF, new_jinsDF)
            # new_df.to_pickle(save_name_str+"_EXP%d_wJINS.pickle"%(experiment_mode))
            # new_df.to_csv(save_name_str+"_EXP%d_wJINS.csv"%(experiment_mode))
            
#            new_imu_df = putTrialinIMU(exp1.trialDF, new_jinsDF)
#            new_imu_df.to_csv(save_name_str+"_EXP%d.csv"%(experiment_mode))
#            new_imu_df.to_pickle(save_name_str+"_EXP%d.pickle"%(experiment_mode))
            print("Result saved")
            myWindow.refreshtrainingDatacomboBox()
        except:
            print("Error on combining EXP/JINS data")
            
            
# call the 'run' method
if __name__ == '__main__':
    # Feed it the flask app instance 
    # ui = FlaskUI(app)
    # ui.run()
    

    checkFolder(save_folder)
    checkFolder(save_trained_folder)


    url = "http://127.0.0.1:5000"

    # threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    app.run()