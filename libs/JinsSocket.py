# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
import threading
import socket
import datetime

import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

class JinsSocket(threading.Thread):
    def __init__(self, isUDP = False, Port = 12562, w_size=2000, save_name="test", save_online=True):
        threading.Thread.__init__(self)
        
        # set your default ip&port  
        # self.IP = socket.gethostbyname(socket.gethostname())
        self.IP = "127.0.0.1"
        self.Port = Port
        self.isUDP = isUDP
        if not self.isUDP:
            self.setLocalIP()
            
        
        """ Define the length of different mode array """
        self.FULL_COUNT = 13
        
        """ Define Array """
        self.w_size = w_size
        
        
#        self.EogH = np.zeros(0)
#        self.EogV = np.zeros(0)     
        self.EogL = np.zeros(self.w_size)
        self.EogR = np.zeros(self.w_size)
        
        self.GyroX = np.zeros(self.w_size)
        self.GyroY = np.zeros(self.w_size)  
        self.GyroZ = np.zeros(self.w_size)
        
        self.AccX = np.zeros(self.w_size)
        self.AccY = np.zeros(self.w_size)  
        self.AccZ = np.zeros(self.w_size)
        
        self.TIME = np.zeros(self.w_size)
        
        self.PAUSE_loop = False
        self.STOP_loop = False
        
        self.counter = 0
        self.last_act_t = 0
        
        """Online Saving"""
        self.save_online = save_online
        if self.save_online:
            self.online_save_file = open(save_name+"_JINS.csv","w") 
            self.online_save_file.write(",EpochTime,EOG_L,EOG_R,EOG_H,EOG_V,GYRO_X,GYRO_Y,GYRO_Z,ACC_X,ACC_Y,ACC_Z\n")
        
        
        """moving average"""
        self.mAverage_num = 100
        
        self.EogL_average = 0
        self.EogR_average = 0
        
        self.GyroX_average = 0
        self.GyroY_average = 0
        self.GyroZ_average = 0
        
        self.AccX_average = 0
        self.AccY_average = 0
        self.AccZ_average = 0
        
        offset = time.timezone if (time.localtime().tm_isdst == 0) else time.altzone
        self.timezone_diff = (offset / 60 / 60 * -1)
        
    def setLocalIP(self):
        self.IP = socket.gethostbyname(socket.gethostname())

    def setIP_Port(self, IP, Port):
        self.IP = IP
        self.Port = Port
        
    def setConnection(self):
        print("CONNECTING...",self.IP,":",self.Port)
#==============================================================================
#         [UDP]setting the socket comunication
#==============================================================================
        if self.isUDP:
            try:
                self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.serverSocket.bind(('', self.Port))
            except:   
                self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.serverSocket.bind(('', self.Port))
#==============================================================================
#         [TCP]setting the socket comunication
#==============================================================================
        else:
            try:
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect((self.IP, self.Port))
            except:
                self.client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect((self.IP, self.Port))
   
    def run(self):
        while True:
            if self.STOP_loop:
                print("loop STOP")
                break
            
            if not self.PAUSE_loop:            
                try:
                    if self.isUDP:
                        data, address = self.serverSocket.recvfrom(2048)
                    else:
                        data = self.client.recv(2048)
                except socket.error as e:
                    print("Socket Error: %s"%e)
                    if self.STOP_loop:
                        print("loop STOP")
                        break
                                    
                converted = data.decode("utf-8")
                lines = converted.split("\r\n")
                for line in lines:
                    if self.isUDP:
                        values = list(map(int, line.split(",")))
                        if len(values) == 9:
                            self.addFULL(values)
                        else:
                            break
                    else:
                        values = line.split(",")
                        if len(values) == 13:
                            
                            vals_dict = {'DATE': values[2],
                                         
                                         'EOG_L': values[9],
                                         'EOG_R': values[10],
                                         
                                         'GYRO_X': values[6],
                                         'GYRO_Y': values[7],
                                         'GYRO_Z': values[8],
                                         
                                         'ACC_X': values[3],
                                         'ACC_Y': values[4],
                                         'ACC_Z': values[5]}
                            self.addFULLbyDict(vals_dict)
                    
                        
    def __del__(self):
        self.close()
              
    def close(self):
        self.STOP_loop = True
        time.sleep(0.5)
        self.Disconnect()
    
    def Disconnect(self):
        if self.isUDP:
            self.serverSocket.close()
        else:
            self.client.close()
        print(">>>>Jins Socket closed")


    """ Full Mode """
    def addFULL(self, full_values):
            
            
        self.TIME = np.roll(self.TIME,-1)
        self.TIME[-1] = full_values[8]
        
        self.EogL = np.roll(self.EogL,-1)
        self.EogL[-1] = full_values[6]
        self.EogR = np.roll(self.EogR,-1)
        self.EogR[-1] = full_values[7]
        
        self.GyroX = np.roll(self.GyroX,-1)
        self.GyroX[-1] = full_values[3]
        self.GyroY = np.roll(self.GyroY,-1)
        self.GyroY[-1] = full_values[4]
        self.GyroZ = np.roll(self.GyroZ,-1)
        self.GyroZ[-1] = full_values[5]
        
        self.AccX = np.roll(self.AccX,-1)
        self.AccX[-1] = full_values[0]
        self.AccY = np.roll(self.AccY,-1)
        self.AccY[-1] = full_values[1]
        self.AccZ = np.roll(self.AccZ,-1)
        self.AccZ[-1] = full_values[2]
        
        if self.save_online:
            self.online_save_file.write(self.getLast_str()+"\n")
            
        self.updateMovingAverage()
            
    def addFULLbyDict(self, full_dict):
            
            
        self.TIME = np.roll(self.TIME,-1)
        self.TIME[-1] = convertJinsDATEtoEpoch(full_dict['DATE'], self.timezone_diff)
        
        self.EogL = np.roll(self.EogL,-1)
        self.EogL[-1] = full_dict['EOG_L']
        self.EogR = np.roll(self.EogR,-1)
        self.EogR[-1] = full_dict['EOG_R']
        
        self.GyroX = np.roll(self.GyroX,-1)
        self.GyroX[-1] = full_dict['GYRO_X']
        self.GyroY = np.roll(self.GyroY,-1)
        self.GyroY[-1] = full_dict['GYRO_Y']
        self.GyroZ = np.roll(self.GyroZ,-1)
        self.GyroZ[-1] = full_dict['GYRO_Z']
        
        self.AccX = np.roll(self.AccX,-1)
        self.AccX[-1] = full_dict['ACC_X']
        self.AccY = np.roll(self.AccY,-1)
        self.AccY[-1] = full_dict['ACC_Y']
        self.AccZ = np.roll(self.AccZ,-1)
        self.AccZ[-1] = full_dict['ACC_Z']
        
        if self.save_online:
            self.online_save_file.write(self.getLast_str()+"\n")
            
        self.updateMovingAverage()
        
        
#        self.TIME = np.append(self.GyroZ,full_values[8])
#        
#        self.EogL = np.append(self.EogL,full_values[6])
#        self.EogR = np.append(self.EogR,full_values[7])
#        
#        self.GyroX = np.append(self.GyroX,full_values[3])
#        self.GyroY = np.append(self.GyroY,full_values[4])
#        self.GyroZ = np.append(self.GyroZ,full_values[5])

#        self.EogH = np.append(self.EogH,full_values[6]-full_values[7])
#        self.EogV = np.append(self.EogV, -(full_values[6]+full_values[7])/2)
#        self.checkSIZE()
    def updateMovingAverage(self):
        self.EogL_average = self.CumulativeAverage(self.EogL_average, self.EogL[-1])
        self.EogR_average = self.CumulativeAverage(self.EogR_average, self.EogR[-1])
        
        self.GyroX_average = self.CumulativeAverage(self.GyroX_average, self.GyroX[-1])
        self.GyroY_average = self.CumulativeAverage(self.GyroY_average, self.GyroY[-1])
        self.GyroZ_average = self.CumulativeAverage(self.GyroZ_average, self.GyroZ[-1])
        
        self.AccX_average = self.CumulativeAverage(self.AccX_average, self.AccX[-1])
        self.AccY_average = self.CumulativeAverage(self.AccY_average, self.AccY[-1])
        self.AccZ_average = self.CumulativeAverage(self.AccZ_average, self.AccZ[-1])
        
        
    def CumulativeAverage(self, cur_average, last_value):
        return (self.mAverage_num*cur_average + last_value)/(self.mAverage_num+1)
        
        
    def checkSIZE(self):
        if len(self.EogL) > self.w_size:
            self.EogL = self.EogL[-self.w_size:]
            self.EogR = self.EogR[-self.w_size:]
            
            self.GyroX = self.GyroX[-self.w_size:]
            self.GyroY = self.GyroY[-self.w_size:]
            self.GyroZ = self.GyroZ[-self.w_size:]
            
            self.AccX = self.AccX[-self.w_size:]
            self.AccY = self.AccY[-self.w_size:]
            self.AccZ = self.AccZ[-self.w_size:]
            
            self.TIME = self.TIME[-self.w_size:]
        else:
            self.Buffer = np.array([0])
    
    def saveTXT(self,f_name='saving_test'):
        str_header = "EpochTime, EogL, EogR, GyroX, GyroY, GyroZ"
        total = np.array([self.TIME, self.EogL, self.EogR, self.GyroX, self.GyroY, self.GyroZ]).T
        np.savetxt("%s_JINS.np"%(f_name),total, header=str_header)
        
        return total
    def loadTXT(self,f_name='saving_test'):
        try:
            loaded = np.loadtxt(f_name)
            self.TIME = loaded[:,0]
            self.EogL = loaded[:,1]
            self.EogR = loaded[:,2]
            self.GyroX = loaded[:,3]
            self.GyroY = loaded[:,4]
            self.GyroZ = loaded[:,5]
            print("JINS: file loaded")
        except:
            print("JINS: error on loading file")
    def saveasPickle(self, name="NoName", column_list=["EpochTime", "EOG_L", "EOG_R", "EOG_H", "EOG_V", "GYRO_X","GYRO_Y" ,"GYRO_Z", "ACC_X", "ACC_Y","ACC_Z" ]):
        total = np.array([self.TIME, self.EogL, self.EogR, (self.EogL-self.EogR), (self.EogL+self.EogR)/2, self.GyroX, self.GyroY, self.GyroZ, self.AccX, self.AccY, self.AccZ]).T
        one_df = pd.DataFrame(total, columns=column_list)
        one_df.to_csv('%s_JINS.csv'%(name))
        one_df.to_pickle('%s_JINS.pickle'%(name))
        
        return one_df
       
        
    def getLast(self,size):
        return self.TIME[-size:], self.EogL[-size:],self.EogR[-size:], self.GyroX[-size:],self.GyroY[-size:],self.GyroZ[-size:]
    def getLast_str(self):
        str_ = ",{},{},{},{},{},{},{},{},{},{},{}".format(self.TIME[-1],                                                          
                                                          self.EogL[-1], self.EogR[-1],
                                                          (self.EogL[-1]-self.EogR[-1]), (self.EogL[-1]+self.EogR[-1])/2,
                                                          self.GyroX[-1], self.GyroY[-1], self.GyroZ[-1],
                                                          self.AccX[-1], self.AccY[-1], self.AccZ[-1])
        return str_

    def getLast_dict_one(self, delta_value=False):
        if delta_value:
            eog_l = self.EogL[-1] - self.EogL_average
            eog_r = self.EogR[-1] - self.EogR_average
            data_dict = {"ACC_X": self.AccX[-1] - self.AccX_average,
                        "ACC_Y": self.AccY[-1] - self.AccY_average,
                        "ACC_Z": self.AccZ[-1] - self.AccZ_average,
                        
                        "GYRO_X": self.GyroX[-1] - self.GyroX_average,
                        "GYRO_Y": self.GyroY[-1] - self.GyroY_average,
                        "GYRO_Z": self.GyroZ[-1] - self.GyroZ_average,
                        
                        "EOG_L": eog_l,
                        "EOG_R": eog_r,
                        "EOG_H": eog_l-eog_r,
                        "EOG_V": (eog_l+eog_r)/2,
                        "TIME": self.TIME[-1]
                        
                            }
                    
            return data_dict
            
        data_dict = {   "ACC_X": self.AccX[-1],
                        "ACC_Y": self.AccY[-1],
                        "ACC_Z": self.AccZ[-1],
                        
                        "GYRO_X": self.GyroX[-1],
                        "GYRO_Y": self.GyroY[-1],
                        "GYRO_Z": self.GyroZ[-1],
                        
                        "EOG_L": self.EogL[-1],
                        "EOG_R": self.EogR[-1],
                        "EOG_H": self.EogL[-1]-self.EogR[-1],
                        "EOG_V": (self.EogL[-1]+self.EogR[-1])/2,
                        "TIME": self.TIME[-1]
                            }
                    
        return data_dict
        
    def getLast_dict(self, size, delta_value=False):
        if delta_value:
            eog_l = self.EogL[-size:] - self.EogL_average
            eog_r = self.EogR[-size:] - self.EogR_average
            data_dict = {"ACC_X": self.AccX[-size:] - self.AccX_average,
                        "ACC_Y": self.AccY[-size:] - self.AccY_average,
                        "ACC_Z": self.AccZ[-size:] - self.AccZ_average,
                        
                        "GYRO_X": self.GyroX[-size:] - self.GyroX_average,
                        "GYRO_Y": self.GyroY[-size:] - self.GyroY_average,
                        "GYRO_Z": self.GyroZ[-size:] - self.GyroZ_average,
                        
                        "EOG_L": eog_l,
                        "EOG_R": eog_r,
                        "EOG_H": eog_l-eog_r,
                        "EOG_V": (eog_l+eog_r)/2,
                        "TIME": self.TIME[-size:]
                        
                            }
                    
            return data_dict
            
        data_dict = {   "ACC_X": list(self.AccX[-size:]),
                        "ACC_Y": list(self.AccY[-size:]),
                        "ACC_Z": list(self.AccZ[-size:]),
                        
                        "GYRO_X": list(self.GyroX[-size:]),
                        "GYRO_Y": list(self.GyroY[-size:]),
                        "GYRO_Z": list(self.GyroZ[-size:]),
                        
                        "EOG_L": list(self.EogL[-size:]),
                        "EOG_R": list(self.EogR[-size:]),
                        "EOG_H": list(self.EogL[-size:]-self.EogR[-size:]),
                        "EOG_V": list((self.EogL[-size:]+self.EogR[-size:])/2),
                        "TIME": list(self.TIME[-size:])
                            }
                    
        return data_dict
    def getLastbyTime_dict(self, dt=1000, delta_value=False):
        cur_t = self.TIME[-1]
        target_t = cur_t - dt
        
        indexes = np.where((self.TIME >= target_t-5) & (self.TIME <= target_t+5))[0]
        if len(indexes)==0:
            i = -100
        else:
            i = indexes[0]
        
        
        if delta_value:
            eog_l = self.EogL[i:] - self.EogL_average
            eog_r = self.EogR[i:] - self.EogR_average
            data_dict = {"ACC_X": self.AccX[i:] - self.AccX_average,
                        "ACC_Y": self.AccY[i:] - self.AccY_average,
                        "ACC_Z": self.AccZ[i:] - self.AccZ_average,
                        
                        "GYRO_X": self.GyroX[i:] - self.GyroX_average,
                        "GYRO_Y": self.GyroY[i:] - self.GyroY_average,
                        "GYRO_Z": self.GyroZ[i:] - self.GyroZ_average,
                        "EOG_L": eog_l,
                        "EOG_R": eog_r,
                        "EOG_H": eog_l-eog_r,
                        "EOG_V": (eog_l+eog_r)/2,
                        "TIME": self.TIME[i:]
                        
                            }
                    
            return data_dict
            
        data_dict = {"ACC_X": self.AccX[i:],
                        "ACC_Y": self.AccY[i:],
                        "ACC_Z": self.AccZ[i:],
                        
                        "GYRO_X": self.GyroX[i:],
                        "GYRO_Y": self.GyroY[i:],
                        "GYRO_Z": self.GyroZ[i:],
                        
                        "EOG_L": self.EogL[i:],
                        "EOG_R": self.EogR[i:],
                        "EOG_H": self.EogL[i:]-self.EogR[i:],
                        "EOG_V": (self.EogL[i:]+self.EogR[i:])/2,
                        "TIME": self.TIME[i:]
                            }
                    
        return data_dict
        
    def getByTime(self, start_time, dt):
        start_from = np.argwhere(start_time < self.TIME)
        to_end = np.argwhere(self.TIME < start_time+dt)
        
        candidate_indexes = np.nonzero(np.in1d(start_from,to_end))[0]
        
        try:
            min_i = candidate_indexes[0]
            max_i = candidate_indexes[-1]+1
            
            return 1,self.TIME[min_i:max_i], self.EogL[min_i:max_i],self.EogR[min_i:max_i], self.GyroX[min_i:max_i],self.GyroY[min_i:max_i],self.GyroZ[min_i:max_i]
        except:
            return -1,-1


def loadJinsCSV(file):
    column_names = ["ARTIFACT" ,"NUM" ,"DATE" ,"ACC_X" ,"ACC_Y" ,"ACC_Z" ,"GYRO_X" ,"GYRO_Y" ,"GYRO_Z" ,"EOG_L" ,"EOG_R" ,"EOG_H" ,"EOG_V"]
    df = pd.read_csv(file, comment='/', names=column_names)
    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    
    start = 0
    for i,line in enumerate(lines):
        if '//' not in line:
            start = i
            break
    my_lst_str = ''.join(map(str, lines[start:]))
    
    df = pd.read_csv(StringIO(my_lst_str), names=column_names)
    df['fileName']=file
    df = df.drop('ARTIFACT', 1)
    
    return df
    
def loadJinsCSVnocomment(file):
    column_names = ["ARTIFACT" ,"NUM" ,"DATE" ,"ACC_X" ,"ACC_Y" ,"ACC_Z" ,"GYRO_X" ,"GYRO_Y" ,"GYRO_Z" ,"EOG_L" ,"EOG_R" ,"EOG_H" ,"EOG_V"]
    df = pd.read_csv(file, names=column_names)
    df['fileName']=file
    df = df.drop('ARTIFACT', 1)
    
    return df
    
def convertJinsDATEtoEpoch(date, time_zone_diff=9, multiply=1000):
    if type(date) == int:
        return date
    
    try:
        jins_pattern = '%Y-%m-%d %H:%M:%S.%f'
        jins_datetime = datetime.datetime.strptime(date, jins_pattern)+datetime.timedelta(hours=time_zone_diff)
    except:
        try:
            jins_pattern = '%Y/%m/%d %H:%M:%S.%f'
            jins_datetime = datetime.datetime.strptime(date, jins_pattern)+datetime.timedelta(hours=time_zone_diff)
        except:
            print("Error on date format: ",date,type(date),"|",int(date))
    
    epoch = int(jins_datetime.timestamp()*multiply)
    
    return epoch
def convertJinsDATEtoEpoch_forAPPLY(df):
    return convertJinsDATEtoEpoch(df['DATE'])

def addEpochonJinsDF(jins_df, time_zone_diff=9, multiply=1000):
    out_df = jins_df.copy()
    out_df['EpochTime']=-1
    # out_df['EpochTime'] = out_df.apply(convertJinsDATEtoEpoch_forAPPLY,args=[time_zone_diff, multiply], axis=1)
    out_df['EpochTime'] = out_df.apply(lambda x: convertJinsDATEtoEpoch(x['DATE'],time_zone_diff, multiply), axis=1)
    
#    dates = out_df['DATE']
#    for i in range(len(out_df)):
#        t = convertJinsDATEtoEpoch(str(dates[i]), time_zone_diff=time_zone_diff)
#        out_df.set_value(i,'EpochTime',t)
        
    return out_df
    #%%

    
#%%    
"""working for retrieve moving average of offline file"""
#==============================================================================
# 
# import pandas as pd
# import numpy as np
# # 'ExperimentData/170710_5+5/',
# folders = [ 'ExperimentData/170710_5+5/', 'ExperimentData/170710_5+5/', 'ExperimentData/170706_5+5/', 
#            'ExperimentData/Jedidiah170510/Sitting/', 'ExperimentData/Jedidiah170510/Walking/',
#            'ExperimentData/sit_old/', 'ExperimentData/walking/', 'ExperimentData/sit_new/',
#            'ExperimentData/walking_new/']
# for folder in folders:
#     print("Converting...: %s"%(folder))
#     jins_df =  pd.read_csv(folder+'Jins_allCombined.csv')
#     new_jinsDF = addMovingAverage(jins_df)
#     
#     new_jinsDF.to_csv(folder+'Jins_allCombined_mAverage.csv',index=False)    
#==============================================================================
    
#%%

def addMovingAverage(jins_df,num=100):
    out_df = jins_df.copy()
    
    epoch = jins_df['EpochTime'].as_matrix()
    d_epoch = np.abs(np.diff(epoch))
    indexes = np.where( d_epoch > 5000 )[0]

    keys = ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z", "EOG_L", "EOG_R", "EOG_H", "EOG_V"]
    mAverage_keys = ['mAverge'+key for key in keys]
    averages = dict()
    for key in mAverage_keys:
        out_df[key] = 0
        averages[key] = 0
    
        
    
    i=0
    while True:
        if i==0 or i in indexes:
            #print(i)
            part_df = out_df.loc[i:i+num-1]
            for ii, key in enumerate(keys):
                averages[mAverage_keys[ii]] = part_df[key].mean()
            i = i+num
        
        else:
            part_df = out_df.loc[i]
            for ii, key in enumerate(keys):
                averages[mAverage_keys[ii]] = CumulativeAverage(averages[mAverage_keys[ii]], part_df[key])
                out_df.set_value(i, mAverage_keys[ii], averages[mAverage_keys[ii]])
            i+=1
            
        
        if i == len(out_df):
            break
    
    return out_df

def CumulativeAverage( cur_average, last_value):
    return (100*cur_average + last_value)/(100+1)



if __name__ == "__main__":

    """Thread 1: DATA COLLECTION """
    jins_client = JinsSocket(isUDP=False, Port=12562, w_size=3000, save_name="testing")
    jins_client.setConnection()
    jins_client.start()
    
    dt = 1
    last_t = 0
    while True:
        print(jins_client.getLast_str()) 
        
        time.sleep(dt)
        
    
    jins_client.close()