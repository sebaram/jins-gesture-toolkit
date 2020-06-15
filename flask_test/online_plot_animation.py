# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 23:44:15 2020

@author: JY
"""

import sys
sys.path.append("../libs")
from sensorUDP import imus_UDP
import JinsSocket
import queue

import time
from matplotlib import pyplot as plt
import numpy as np
from pynput import keyboard
import pandas as pd
from threading import Thread

current_milli_time = lambda: int(round(time.time() * 1000))


jins_client = JinsSocket.JinsSocket(isUDP=True, Port=12562, w_size=100*60*5)
jins_client.setConnection()
jins_client.start()
jins_data_q = queue.Queue(maxsize=10)


# TARGET_IMU_IP = '192.168.0.12'
TARGET_IMU_IP = '192.168.0.186'

imu_get = imus_UDP(Port=12563)
imu_get.setConnection()
imu_get.start()
imu_data_q = queue.Queue(maxsize=10)
#%%

STOP_LOOP = False

a_pressed = False
def on_press(key):
    global STOP_LOOP, a_pressed, pressed_df, jins_client, imu_get
    try:
        if key.char == 'a' and not a_pressed:
            a_pressed = True
            
            pressed_df.loc[len(pressed_df)] = [key.char, 'pressed',
                                               jins_client.getLast_dict_one()['TIME'],
                                               imu_get.getDATA(1)[TARGET_IMU_IP][-1,11]]
        
        # print('alphanumeric key {0} pressed'.format(key.char))
    except AttributeError:
        _tmp = 1
        # print('special key {0} pressed'.format(key))

def on_release(key):
    global STOP_LOOP, a_pressed, pressed_df, anim
    # print('{0} released'.format(
    #     key))
    
    try:
        if key.char == 'a' and a_pressed:
            a_pressed = False
            pressed_df.loc[len(pressed_df)] = [key.char, 'released',
                                               jins_client.getLast_dict_one()['TIME'],
                                               imu_get.getDATA(1)[TARGET_IMU_IP][-1,11]]
        elif key.char == '1':
            print("run device sync")
    except AttributeError:
        if key == keyboard.Key.esc:
            STOP_LOOP = True
            terminate_all()
            
            
            # Stop listener
            return False

    
    
pressed_df = pd.DataFrame(columns=['key', 'event', 'jins_time', 'imu_time'])

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(on_press=on_press,
                             on_release=on_release)
listener.start()


PRESS_NUM = 200
pressed_log = np.zeros(PRESS_NUM)
def press_counter(threadname):
    global STOP_LOOP, a_pressed, pressed_log
    
    while True:
        if STOP_LOOP:
            break
        
        pressed_log = np.roll(pressed_log, -1)
        if a_pressed:
            pressed_log[-1] = 1
        else:
            pressed_log[-1] = 0
        time.sleep(0.02)
thread1 = Thread( target=press_counter, args=("Thread-1", ) )
thread1.start()
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, axes = plt.subplots(5,1, figsize=(10,10))

def terminate_all():
    global ani, jins_client, imu_get
    try:
        ani.event_source.stop()
    except:
        print("cannot detect 'anim'")    
    
    jins_client.close()
    imu_get.close()
    # plt.close()


JINS_NUM = 400
JINS_X = np.arange(JINS_NUM)
jins_client.getLast_dict(JINS_NUM, q=jins_data_q)
jins_data = jins_data_q.get()
jins_data_q.task_done()

jins_lines = dict()
for key, val in jins_data.items():
    if key in ["EOG_L","EOG_R","EOG_H","EOG_V"]:
        line, = axes[0].plot(val, lw=3)
        
        jins_lines[key] = (0, line)
    if key in ["GYRO_X","GYRO_Y","GYRO_Z"]:
        line, = axes[2].plot(val, lw=3)
        
        jins_lines[key] = (2, line)
axes[0].set_xlim(0,JINS_NUM)
axes[0].set_ylim([-2000,2000])
axes[0].set_title("Jins EOG")

axes[2].set_xlim(0,JINS_NUM)
axes[2].set_ylim([-36000,36000])
axes[2].set_title("Jins Gyro")





IMU_NUM = 200
IMU_X = np.arange(IMU_NUM)

imu_get.getDATA(IMU_NUM, imu_data_q)
imu_data = imu_data_q.get()[TARGET_IMU_IP]
imu_data_q.task_done()

imu_lines = dict()
line, = axes[1].plot(imu_data[:,13], lw=3)
imu_lines['MagX'] = (1,line)
line, = axes[1].plot(imu_data[:,14], lw=3)
imu_lines['MagY'] = (1,line)
line, = axes[1].plot(imu_data[:,15], lw=3)
imu_lines['MagZ'] = (1,line)


line, = axes[3].plot(imu_data[:,1], lw=3)
imu_lines['Gx'] = (3,line)
line, = axes[3].plot(imu_data[:,2], lw=3)
imu_lines['Gy'] = (3,line)
line, = axes[3].plot(imu_data[:,3], lw=3)
imu_lines['Gz'] = (3,line)


axes[1].set_xlim(0,IMU_NUM)
axes[1].set_ylim([-1000,1000])
axes[1].set_title("Watch MAG")

axes[3].set_xlim(0,IMU_NUM)
axes[3].set_ylim([-20,20])
axes[3].set_title("Watch Gyro")



pressed_line, = axes[4].plot(pressed_log, lw=3)
axes[4].set_xlim(0,IMU_NUM)
axes[4].set_ylim([0,1.2])
axes[4].set_title("Pressed")


text = axes[4].text(0.8,0.5, "")
cur_time = 0
def update(frame, *factor):
    global cur_time
    jins_client, imu_get = factor[0], factor[1]
    
    
    jins_data = jins_client.getLast_dict(JINS_NUM)
    
    for key, val in jins_lines.items():
        val[1].set_data(JINS_X, jins_data[key])
    
    new_time = current_milli_time()
    
    pressed_line.set_data(IMU_X, pressed_log)
    
    new_time = current_milli_time()
    text.set_text("dt:{:03}ms |a_pressed: {}".format(new_time-cur_time,a_pressed))
    cur_time = new_time
    
    imu_data = imu_get.getDATA(IMU_NUM)[TARGET_IMU_IP]
    imu_lines['MagX'][1].set_data(IMU_X, imu_data[:,13])
    imu_lines['MagY'][1].set_data(IMU_X, imu_data[:,14])
    imu_lines['MagZ'][1].set_data(IMU_X, imu_data[:,15])
    imu_lines['Gx'][1].set_data(IMU_X, imu_data[:,1])
    imu_lines['Gy'][1].set_data(IMU_X, imu_data[:,2])
    imu_lines['Gz'][1].set_data(IMU_X, imu_data[:,3])
    
    artists = [one_[1] for one_ in imu_lines.values()] +\
                [one_[1] for one_ in jins_lines.values()] +\
                [text, pressed_line]
    return tuple(artists)

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    fargs=(jins_client, imu_get),
                    interval=10,
                    blit=True)
plt.show(block=False)