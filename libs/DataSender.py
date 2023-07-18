#%%
import socket
import struct
import time
current_millis = lambda: int(round(time.time() * 1000))

class DataSender():
    def __init__(self, IP:str, Port:int, dataFormat='qffffffffff'):
        self.IP = IP
        self.Port = Port

        self.dataFormat = dataFormat
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.last_direction_sent = -1
        self.last_direction = ""
        self.last_bezel_side = ""
    def sendData(self, data_list:list):
        if len(data_list) != len(self.dataFormat)-1:
            print("Data length is not matched with dataFormat: get:{}, expected:{}".format(len(data_list), len(self.dataFormat)-1))
            return

        full_list = data_list + [current_millis()]
        full_list.reverse()
        data = struct.pack(self.dataFormat, *full_list)
        arr = bytearray(data)
        arr.reverse()
        self.serverSocket.sendto(arr, (self.IP, self.Port))
    def sendDirection(self, direction:str, bezel_side:str="in"):
        if self.last_direction == direction and self.last_bezel_side==bezel_side and current_millis()-self.last_direction_sent < 1000:
            return

        data_list = [23]+[-99,-99,-99,-99,-99,-99,-99,-99,-99]

        data_list[3] = 1 if bezel_side=="in" else -1   # inside-out
        if direction=="left":
            data_list[1] = -1
            data_list[2] = 0
        elif direction=="right":
            data_list[1] = 1
            data_list[2] = 0
        elif direction=="up":
            data_list[1] = 0
            data_list[2] = 1
        elif direction=="down":
            data_list[1] = 0
            data_list[2] = -1
        else:
            print("Wrong direction: {}".format(direction))
            return
        self.sendData(data_list)

        self.last_direction_sent = current_millis()
        self.last_direction = direction
        self.last_bezel_side = bezel_side

if __name__ == "__main__":

    data_sender = DataSender("192.168.0.67",11563)
    float_list = [i for i in range(10)]
    data_sender.sendData(float_list)

    import time
    time.sleep(3)

    data_sender.sendDirection("left")

