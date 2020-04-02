# -*- coding: utf-8 -*-
import time

import numpy as np
import pandas as pd

import pygame

from playsound import playsound
from pathlib import Path
from gtts import gTTS

import os

# set Print strings

                  
"""Functions"""
current_milli_time = lambda: int(round(time.time() * 1000))
current_milli_time_f = lambda: float(time.time() * 1000)



from array import array
from pygame.mixer import Sound, get_init

class Note(Sound):

    def __init__(self, frequency, volume=.1):
        self.frequency = frequency
        Sound.__init__(self, self.build_samples())
        self.set_volume(volume)

    def build_samples(self):
        period = int(round(get_init()[0] / self.frequency))
        samples = array("h", [0] * period)
        amplitude = 2 ** (abs(get_init()[1]) - 1) - 1
        for t in range(period):
            if t < period / 2:
                samples[t] = amplitude
            else:
                samples[t] = -amplitude
        return samples
"""
experiment mode = 1: auto time count, 2:wait till succeed, 3:False/Positive recording mode

current_state = 0:waiting 1:count down 2:recording  3:all trial done
"""
class Experiment:
    def __init__(self, experiment_mode, 
                 false_action = ["reading", "talking", "computer"], 
                 name="unamed", trial_num=10, 
                 size=[1920,1080],
                 typeText = {0: "Left Flick ",
                             1: "Left Push  ",
                             2: "Right Flick",
                             3: "Right Push ",
                             4: "Rubbing    "},
                 tts = True,
                 typeSound = { 0: "sound_files/Left_Flick.mp3",
                              1: "sound_files/Left_Push.mp3",
                              2: "sound_files/Right_Flick.mp3",
                              3: "sound_files/Right_Push.mp3",
                              4: "sound_files/Rubbing.mp3"},
                 mp3_save_folder = "mp3",
                 total_time = 5, random_order = True):
        self.size = size
        self.size_ratio = (size[0]/1920)
        
        self.type_num = len(typeText)
        self.trial_num = trial_num
        self.typeText = typeText
        self.random_order = random_order
        self.current_target = -1
        
        if not os.path.exists(mp3_save_folder):
            os.makedirs(mp3_save_folder)
        

        self.enable_tts = tts
        if self.enable_tts:
            tmp_sounds = dict()
            
            for onekey in typeText.keys():
                f_name = mp3_save_folder + "/" +typeText[onekey] + ".mp3"
                if Path(f_name).exists():
                    print(f_name," exists. Using exist file")

                else:
#                    tts = gTTS(text=typeText[onekey], lang='ko')
                    tts = gTTS(text=typeText[onekey], lang='en')
                    tts.save(f_name)
                tmp_sounds[onekey] = f_name
            self.typeSound = tmp_sounds
        else:
            self.typeSound = typeSound
        
        self.tot_trials = self.type_num * self.trial_num
        self.start_time = current_milli_time()
        self.trial_index = -1
        self.targetTypeDirection = -999
        self.inputTypeDirection = -999
        self.trial_count = self.trial_num*np.ones(self.type_num)
        self.allTrial_done = False
        
        self.trial_succeed = False
        self.last_name = "None"
        self.last_accuracy = 0.0

        
        self.recording = False
        self.NAME = name
        self.rollBacked_already = False
                              
        self.false_actions = false_action
        self.falseInput_done = False
        self.false_index = 0
        self.false_NUM = len(self.false_actions)
                              
        self.initialColumn = ['TrialNum', 'Name',
                              'Recording', 'StartTime', 'CurrentTime',
                              'TargetName', 'Target', 'Result', 'Accuracy',
                              'FalseIndex', 'FalseActions']
        self.trialDF = pd.DataFrame(columns=self.initialColumn+['DATA'])
        
        self.DATAColumn = ['TrialNum', 'Result']
        self.DATA_DF = pd.DataFrame(columns=self.DATAColumn)
        self.string_instruction = "CURRENT: -"
        self.string_gesture = "TARGET: -"
        self.color_instruction = (0,100,255)
        self.color_gesture =  (255,100,0)
        
        
        # Font Settings
        self.font_up_size, self.font_up_position = int(100*self.size_ratio), (self.size[0]/2, int(50*self.size_ratio))
        self.font_down_size, self.font_down_position = int(60*self.size_ratio), (self.size[0]/2, int(150*self.size_ratio))
        self.generateFont()
        
        # self.font_up = pygame.font.SysFont("arial", self.font_up_size, bold=True)
# #        self.font_down = pygame.font.SysFont("NotoSansMonoCJKkr-Regular.ttf", int(80*self.size_ratio))
# #        self.font_down = pygame.font.Font("NotoSansMono.otf", int(80*self.size_ratio))
        # try:
            # self.font_down = pygame.font.Font("test.ttf", self.font_down_size)
        # except:
            # self.font_down = pygame.font.SysFont("hack", self.font_down_size)
        
        self.experiment_mode = experiment_mode 
        self.experiment_ongoing = False
        self.total_time = total_time * 60
        self.gap_counter = self.total_time / (self.tot_trials+1)
        
        
        """MODE 1 part"""
    def setDataCollection(self, time_before=3, time_recording=3, total_min = 30):
        self.pre_counter = time_before
        self.record_counter = time_recording
        
        self.total_time = total_min * 60
        self.gap_counter = self.total_time / (self.tot_trials+1)
        
        self.rest_time = -1
        self.current_state = 0 
    def generateFont(self):
        self.font_up = pygame.font.SysFont("arial", self.font_up_size, bold=True)
#        self.font_down = pygame.font.SysFont("NotoSansMonoCJKkr-Regular.ttf", int(80*self.size_ratio))
#        self.font_down = pygame.font.Font("NotoSansMono.otf", int(80*self.size_ratio))
        try:
            self.font_down = pygame.font.Font("test.ttf", self.font_down_size)
        except:
            self.font_down = pygame.font.SysFont("hack", self.font_down_size)
        
        
        
    def startRecording(self):
        if self.current_state==0:
            if self.experiment_mode == 4 and not self.experiment_ongoing:
                self.experiment_ongoing = True
                self.start_time = current_milli_time()
                
                self.trialDF.at[len(self.trialDF),self.initialColumn] = [-999, self.NAME,
                                self.recording, self.start_time, current_milli_time(),
                                'StartPoint', -999, self.last_name, self.last_accuracy, 
                                self.false_index, self.false_actions[self.false_index]]
                self.getNewTrial()
                
                
            else:
                self.current_state = 1
                # Play target gesture sound after press right arrow

                if self.enable_tts and self.targetTypeDirection in self.typeSound.keys():
                    playsound(self.typeSound[self.targetTypeDirection])
                    self.start_time = current_milli_time() #set start_time after play sound
                else:
                    self.start_time = current_milli_time()

    def rollbackOneTrial(self):
        if self.recording:
            return 0
            
        if self.false_index == 0:
            if len(self.trialDF) == 0 or self.rollBacked_already:
                return
            lastTarget = self.trialDF.loc[len(self.trialDF)-1]['Target']
            lastTrialNUM = self.trialDF.loc[len(self.trialDF)-1]['TrialNum']
            
            self.trialDF.at[self.trialDF['TrialNum']==lastTrialNUM, ['TrialNum', 'Result']] = [-1, 'CanceledTrial']
            
            self.trial_count[lastTarget] += 1
            self.trial_index -= 1
            print("Trial NUM:%d removed"%(lastTrialNUM))
            print(self.trial_count)
            self.rollBacked_already = True
    
    def checkStateChange(self):
        # self.current_state 
        # 0:waiting 1:count down 2:recording  3:all trial done
        if self.current_state == 0:
            if self.experiment_mode == 4 and self.experiment_ongoing:
                past_time = (current_milli_time()-self.start_time)/1000
                if past_time >= self.gap_counter:
                    self.startRecording()
                    return True
                else:
                    return False
        
        elif self.current_state == 1:
                past_time = (current_milli_time()-self.start_time)/1000
                self.rest_time = self.pre_counter-past_time
                if past_time >= self.pre_counter:
                    Note(400).play(200)
                    self.recording = True
                    self.last_name = "None"
                    self.last_accuracy = 0.0
                    self.rollBacked_already = False
                    self.start_time = current_milli_time()
                    self.current_state = 2
                    return True
                else:
                    return False
        elif self.current_state == 2:
            if self.experiment_mode in [1,4]:
                """Automatic timer mode"""
                past_time = (current_milli_time()-self.start_time)/1000
                self.rest_time = self.record_counter-past_time
                if past_time >= self.record_counter:
                    self.current_state = 0
                    Note(100).play(50)
                    self.recording = False
                    self.addData()
                    self.getNewTrial()
                    return True
                else:
                    return False
                        
            elif self.experiment_mode == 2:
                """Keep moode till target and performed input matched + Timer"""
                past_time = (current_milli_time()-self.start_time)/1000
                self.rest_time = self.record_counter-past_time
                
                if past_time >= self.record_counter or self.trial_succeed :
                    self.current_state = 0
                    Note(100).play(50)
                    self.recording = False
                    self.addData()
                    self.getNewTrial()
                    return True
                else:
                    return False
                        
            elif self.experiment_mode == 3:
                """False/Positive recording mode"""
                past_time = (current_milli_time()-self.start_time)/1000
                self.rest_time = ""
                if not self.recording:
                    self.current_state = 0
                    self.addData()
                    self.getNewTrial()
                    return True
                else:
                    return False
            
        
    def drawPyGame1(self, screen):
    
            
        # idle
        if self.current_state == 0:
            if self.experiment_mode == 4:
                if self.experiment_ongoing:
                    self.string_instruction = u"wait"
                    self.color_instruction = (100,100,100)
                else:
                    self.string_instruction = u"Press Button to Start"
                    self.color_instruction = (230,230,230)
                
            else:
                self.string_instruction = u"Press Button to Continue"
                self.color_instruction = (0,100,255)
                
        # Countdown
        elif self.current_state == 1:
            self.string_instruction = u"{:02.1f}".format( self.rest_time )
            self.color_instruction = (255,100,100)
            
        # Recording
        elif self.current_state == 2:
            self.string_instruction = u"Now Recording: {:02.1f}".format( self.rest_time )
            self.color_instruction = (0,255,100)
            
        # All done
        elif self.current_state == 3:
            self.string_instruction = u"Finish"
        else:
            self.color_instruction = (100,100,100)
        
        if not [self.current_state, self.experiment_mode] in [[1, 4]]:
            label = self.font_up.render(self.string_instruction, 1, self.color_instruction)
            text_rect = label.get_rect(center=self.font_up_position  )
#            text_rect = label.get_rect(center=(self.size[0]/2, self.size[1]/2 - int(50*self.size_ratio)))
            
            screen.blit(label, text_rect)
        
        if not [self.current_state, self.experiment_mode] in [[0, 4]]:
            label2 = self.font_down.render(self.string_gesture, 1, self.color_gesture)
            text_rect2 = label2.get_rect(center=self.font_down_position)
#            text_rect2 = label2.get_rect(center=(self.size[0]/2, self.size[1]/2 + int(50*self.size_ratio)))
            screen.blit(label2, text_rect2)
        
        
    def getNewTrial(self):
        if self.allTrial_done:
            return -999
        
        self.trial_succeed  = False
        if self.experiment_mode in [1, 2, 4]:
            if np.sum(self.trial_count)==0:
                self.current_state = 3
                self.allTrial_done = True
                
                self.trialDF.at[len(self.trialDF),self.initialColumn] = [-999, self.NAME,
                            self.recording, current_milli_time(), current_milli_time(),
                            'EndPoint', -999, self.last_name, self.last_accuracy, 
                            self.false_index, self.false_actions[self.false_index]]
                
                # set text to show
                self.string_gesture =  u"ALL Finished"
                self.color_gesture = (100,100,100)
            else:
                if self.random_order:
                    while True:
                        cand = np.random.randint(self.type_num)
                        if self.trial_count[cand]>0:
                            self.trial_count[cand] -= 1
                            self.targetTypeDirection = cand
                            
                            break
                else:
                    cand = int((self.trial_index+1)/self.trial_num)
                    self.trial_count[cand] -= 1
                    self.targetTypeDirection = cand
                    
                # set text to show
                self.start_time = current_milli_time()
                self.trial_index += 1
                self.string_gesture =  u"TARGET: {}".format( self.typeText[self.targetTypeDirection] )
                self.color_gesture = (255,100,0)
                
            
            print(self.trial_count)
            
        elif self.experiment_mode == 3:
            self.trial_index += 1
            self.false_index += 1
            
            # set text to show
            self.string_gesture =  u"TARGET: {}".format(self.false_actions[self.false_index])
            if self.false_index >=self.false_NUM:
                self.false_index -= 1
                self.current_state = 3
                
                # set text to show
                self.string_gesture =  u"ALL Finished"
        
        return self.targetTypeDirection
    
    
    def countDetectionResult(self, inputTypeDirection):
        self.inputTypeDirection = inputTypeDirection
		
        if inputTypeDirection == self.targetTypeDirection:
            self.trial_succeed = True
        
        if self.trial_index == -1:
            if self.experiment_mode in [1,2,3]:
                self.getNewTrial()
            return self.targetTypeDirection
            
#        if self.recording:
#            if inputTypeDirection[0]!=0 and inputTypeDirection[1]!=0 :
#                self.trialDF.loc[len(self.trialDF)] = [self.trial_index, self.NAME, self.recording, self.start_time, current_milli_time(), self.typeText[self.targetTypeDirection], self.targetTypeDirection, inputTypeDirection, self.false_index, self.false_actions[self.false_index]]
#            if inputTypeDirection[0] == self.targetTypeDirection[0] and inputTypeDirection[1] == self.targetTypeDirection[1]:
#                self.getNewTrial()
                
        return self.targetTypeDirection
    def countTrialSuccess(self, does_success):
        self.trial_succeed = does_success
        
        if self.trial_index == -1:
            if self.experiment_mode in [1,2,3]:
                self.getNewTrial()
            return self.targetTypeDirection
            
                
        return self.targetTypeDirection
        
    
        
    def drawPyGame2(self, screen, target_type, int_type):
        if target_type != -1:
            self.string_gesture =  "TARGET: " + self.typeText[target_type]
            self.color_gesture = (255,100,0)
            
        if int_type != -1:
            self.string_instruction = "CURRENT: " + self.typeText[int_type]
            self.color_instruction = (0,100,255)
        else:
            self.color_instruction = (100,100,100)
        
        label = self.font_up.render(self.string_instruction, 1, self.color_instruction)
        text_rect = label.get_rect(center=(self.size[0]/2, self.size[1]/2))
        screen.blit(label, text_rect)
        
        label2 = self.font_down.render(self.string_gesture, 1, self.color_gesture)
        text_rect2 = label2.get_rect(center=(self.size[0]/2, self.size[1]/2))
        text_rect2[1] += 100
        screen.blit(label2, text_rect2)
        
    def addData(self):
        self.trialDF.at[len(self.trialDF),self.initialColumn] = [self.trial_index, self.NAME,
                                self.recording, self.start_time, current_milli_time(),
                                self.typeText[self.targetTypeDirection], self.targetTypeDirection, self.last_name, self.last_accuracy, 
                                self.false_index, self.false_actions[self.false_index]]
