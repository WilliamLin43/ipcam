# -*- coding: utf-8 -*-
import cv2 #load opencv
import numpy as np
import time
import pandas as pd
import os
import configparser

fp = pd.read_csv("ipclist.csv")#read data
DEVICE_ID = fp.DEVICE_ID #Device ID
DEVICE_NUMBER = fp.DEVICE_NUMBER #Device Number
IP = fp.IP #IP address data
PORT = fp.PORT #Port data
USER = fp.USER #User data
PWD = fp.PWD #Password data


# Define and parse input arguments
config=configparser.ConfigParser()
config.read("config.txt")    
READ_TIMER = int(config["DEFAULT"]["READ_TIMER"])#args. Image read timer in second

print(DEVICE_ID)
print(DEVICE_NUMBER)
print(IP)
print(PORT)
print(USER)
print(PWD)
print('Reading timer :'+str(READ_TIMER)+'Seconds')

while(True):

    for i in range(len(IP)):
        print("http://"+str(IP[i])+"/media/:"+str(PORT[i])+"?user="+str(USER[i])+"&pwd="+str(PWD[i])+"&action=stream")
        camera = cv2.VideoCapture('http://'+str(IP[i])+'/media/:'+str(PORT[i])+'?user='+str(USER[i])+'&pwd='+str(PWD[i])+'&action=stream')
        
        #read images from test video
        if camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                print(str(DEVICE_ID[i])+','+str(DEVICE_NUMBER[i])+','+str(IP[i])+','+str(PORT[i])+','+str(USER[i])+','+str(PWD[i])+': Connection failed.')
                break
            else:
                img = frame.copy()
                folderpath = "./InputImage/"+str(DEVICE_ID[i])+"_"+str(DEVICE_NUMBER[i]) # 檢查檔案是否存在
                if os.path.isdir(folderpath):
                    print("Folder exist.")
                else:
                    print("Folder not exist.")
                    os.makedirs(folderpath)
                
                fileName=time.strftime("%Y%m%d%H%M%S", time.localtime())
                print(str(folderpath)+'/'+str(fileName)+'.png')
                cv2.imwrite(folderpath+'/'+str(fileName)+'.png',img) #save image
                
                
                #resImg = cv2.resize(img,(300,200),interpolation=cv2.INTER_LINEAR)
                '''
                cv2.namedWindow('Input IP Camera stream', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Input IP Camera stream', 800, 500)
                cv2.imshow("Input IP Camera stream", np.hstack([img])) #show frame & mark cut area
                '''
        else:
            print(str(DEVICE_ID[i])+','+str(DEVICE_NUMBER[i])+','+str(IP[i])+','+str(PORT[i])+','+str(USER[i])+','+str(PWD[i])+': Connection failed.')
        
        camera.release()        
    
    time.sleep(READ_TIMER) #delay a little bit, if require
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()