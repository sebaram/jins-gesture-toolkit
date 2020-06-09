# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:10:40 2020

@author: kctm
"""
import sys
sys.path.append("../libs")
from sensorUDP import imus_UDP
import JinsSocket
import queue

jins_client = JinsSocket.JinsSocket(isUDP=True, Port=12562, w_size=100*60*5)
jins_client.setConnection()
jins_client.start()

imu_get = imus_UDP(Port=12563)
imu_get.setConnection()
imu_get.start()
imu_data_q = queue.Queue(maxsize=100)

#%%
import time
from matplotlib import pyplot as plt
import numpy as np
import threading



fig, axes = plt.subplots(4,1, figsize=(10,10))
axes[0].has_been_closed = False

def on_close(event):
    event.canvas.figure.axes[0].has_been_closed = True
    print("Figure closed")
def on_resize(event):
    global axbackgrounds
    
    axbackgrounds = []
    for one_ax in axes:
        axbackgrounds.append(event.canvas.copy_from_bbox(one_ax.bbox))
fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('resize_event', on_resize)



JINS_NUM = 400
JINS_X = np.arange(JINS_NUM)
jins_data = jins_client.getLast_dict(JINS_NUM)

jins_lines = dict()
for key, val in jins_data.items():
    if key in ["EOG_L","EOG_R","EOG_H","EOG_V"]:
        line, = axes[0].plot(val, lw=3)
        
        jins_lines[key] = (0, line)
    if key in ["GYRO_X","GYRO_Y","GYRO_Z"]:
        line, = axes[2].plot(val, lw=3)
        
        jins_lines[key] = (2, line)
axes[0].set_xlim(0,JINS_NUM)
axes[0].set_ylim([-36000,36000])

axes[2].set_xlim(0,JINS_NUM)
axes[2].set_ylim([-36000,36000])




TARGET_IMU_IP = '192.168.0.12'
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
imu_lines['Gx'] = (1,line)
line, = axes[3].plot(imu_data[:,2], lw=3)
imu_lines['Gy'] = (1,line)
line, = axes[3].plot(imu_data[:,3], lw=3)
imu_lines['Gz'] = (1,line)


axes[1].set_xlim(0,IMU_NUM)
axes[1].set_ylim([-4000,4000])
axes[3].set_xlim(0,IMU_NUM)
axes[3].set_ylim([-20,20])




text = axes[1].text(0.8,0.5, "")

fig.canvas.draw()   # note that the first draw comes before setting data 

# cache the background
on_resize(fig)
plt.show(block=False)
dt=0.01

while True:
    if axes[0].has_been_closed:
        jins_client.close()
        imu_get.close()
        break
    
    jins_data = jins_client.getLast_dict(JINS_NUM)
    for key, val in jins_lines.items():
        val[1].set_data(JINS_X, jins_data[key])
    
    
    imu_get.getDATA(IMU_NUM, imu_data_q)
    imu_data = imu_data_q.get()[TARGET_IMU_IP]
    imu_data_q.task_done()

    imu_lines['MagX'][1].set_data(IMU_X, imu_data[:,13])
    imu_lines['MagY'][1].set_data(IMU_X, imu_data[:,14])
    imu_lines['MagZ'][1].set_data(IMU_X, imu_data[:,15])
    imu_lines['Gx'][1].set_data(IMU_X, imu_data[:,1])
    imu_lines['Gy'][1].set_data(IMU_X, imu_data[:,2])
    imu_lines['Gz'][1].set_data(IMU_X, imu_data[:,3])

    # text.set_text("testing")
    #print tx

    # restore background
    for one_back in axbackgrounds:
        fig.canvas.restore_region(one_back)

    # redraw just the points
    for one_line in jins_lines.values():
        axes[one_line[0]].draw_artist(one_line[1])
    for one_line in imu_lines.values():
        axes[one_line[0]].draw_artist(one_line[1])

    # fill in the axes rectangle
    for one_ax in axes:
        fig.canvas.blit(one_ax.bbox)

    fig.canvas.flush_events()
    #alternatively you could use
    #plt.pause(0.000000000001) 
    # however plt.pause calls canvas.draw(), as can be read here:
    #http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
    time.sleep(dt)
        
