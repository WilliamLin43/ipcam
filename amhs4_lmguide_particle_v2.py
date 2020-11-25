# -*- coding: utf-8 -*-
import cv2 #load opencv
import numpy as np
from numpy import genfromtxt
from PIL import Image, ImageEnhance, ImageFilter
from skimage.morphology import remove_small_objects
import amhs4_lmguide_value2
import pandas as pd
import os
import configparser
import time
import shutil

fp = pd.read_csv("ipclist.csv")#read data
DEVICE_ID = fp.DEVICE_ID #Device ID
DEVICE_NUMBER = fp.DEVICE_NUMBER #Device Number
IP = fp.IP #IP address data
PORT = fp.PORT #Port data
USER = fp.USER #User data
PWD = fp.PWD #Password data
Set_VALUE1=fp.VALUE1
Set_VALUE2=fp.VALUE2

Delay_Time=2

#cut image sub program
def croppoly(img,p):
    #global points
    pts=np.asarray(p)
    pts = pts.reshape((-1,1,2))
    ##Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.fillPoly(mask,pts=[pts],color=(255,255,255))
    #cv2.copyTo(cropped,mask)
    dst=cv2.bitwise_and(cropped,cropped,mask=mask)
    bg=np.ones_like(cropped,np.uint8)*255 #fill the rest with white
    cv2.bitwise_not(bg,bg,mask=mask)
    dst2=bg+dst
    return dst2



while(True):
    
    for i in range(len(IP)):
        folderpath = "./InputImage/"+str(DEVICE_ID[i])+"_"+str(DEVICE_NUMBER[i]) # 檢查檔案是否存在
        print(folderpath)
        if os.path.isdir(folderpath):
            print(str(DEVICE_ID[i])+"_"+str(DEVICE_NUMBER[i])+" Input folder exist.")
            files= os.listdir(folderpath) #得到資料夾下的所有檔名稱
            #print(files)
            for file in files: #遍歷資料夾
                Alarm_Flag=False
                if not os.path.isdir(file): #判斷是否是資料夾,不是資料夾才打開

                    #filename = cv2.imread("./defect2.png")
                    filename = cv2.imread(folderpath+'/'+file)
                    print('Loading :'+str(folderpath)+'/'+str(file))
                    img=filename
                    
                    for j in range(2):
                    
                        #read points record from file
                        dataPath=r'./cutpoints'+str(j)+'.csv'
                        points=genfromtxt(dataPath,delimiter=',').astype(int).tolist()
                        
                        #cut image & save log
                        CheckArea=croppoly(img,points)
                        #cv2.imwrite("./cut"+str(i)+".png", CheckArea) #Save log image
                        #cv2.imshow("CheckArea"+str(i), np.hstack([CheckArea])) #show drawing
                        #CheckArea1=cv2.cvtColor(CheckArea, cv2.COLOR_BGR2GRAY) #轉灰階
                   
                        pts=np.asarray(points)
                        pts = pts.reshape((-1,1,2)) 
                        img2=cv2.polylines(img, [pts], True, (0, 0, 255), 3)
                               
                        
                        cv2.namedWindow('Particle Detection Image & Area', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Particle Detection Image & Area', 1000, 500)
                        cv2.imshow("Particle Detection Image & Area", np.hstack([img2])) #show drawing
                        
                        '''
                        thresh0 = cv2.threshold(CheckArea, 25, 255, cv2.THRESH_BINARY)[1]
                        thresh0 = cv2.erode(thresh0, None, iterations=2)
                        
                        img0 = Image.fromarray(cv2.cvtColor(thresh0,cv2.COLOR_BGR2RGB)) #OpenCV to PIL.Image
                        img0 = img0.filter(ImageFilter.ModeFilter(3))
                        #img0 = img0.filter(ImageFilter.ModeFilter(3))
                        img0 = img0.filter(ImageFilter.EDGE_ENHANCE_MORE)
                        img0 = img0.filter(ImageFilter.MaxFilter(5))
                        
                        #img0 = img0.filter(ImageFilter.MinFilter(3))
                        img0 = cv2.cvtColor(np.array(img0),cv2.COLOR_RGB2BGR) #PIL.Image to OpenCV
                        
                        
                        gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #轉灰階
                        
                        #30 and 150 is the threshold, larger than 150 is considered as edge,
                        #less than 30 is considered as not edge
                        canny = cv2.Canny(gray, 60, 170)
                        canny = np.uint8(np.absolute(canny))
                        ret, binary = cv2.threshold(canny, 60, 150, cv2.THRESH_OTSU+cv2.THRESH_BINARY)       
                        
                        
                        cv2.namedWindow('Particle Detection Area'+str(i), cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Particle Detection Area'+str(i), 800, 200)
                        cv2.imshow("Particle Detection Area"+str(i), np.hstack([CheckArea1,gray,canny,binary])) #show drawing
                        '''
                        
                        Initial_Img = cv2.imread("./cut"+str(j)+"-0.png") #Read initial image
                        frameDelta = cv2.absdiff(Initial_Img, CheckArea)
                        #cv2.imshow("Compare with initial image"+str(i), np.hstack([frameDelta])) #show drawing
                        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
                        thresh = cv2.dilate(thresh, None, iterations=2)
                        #cv2.imshow("Dilate image"+str(i), np.hstack([thresh])) #show drawing
                        
                        img0 = Image.fromarray(cv2.cvtColor(thresh,cv2.COLOR_BGR2RGB)) #OpenCV to PIL.Image
                        img0 = img0.filter(ImageFilter.ModeFilter(11))
                        img0 = img0.filter(ImageFilter.ModeFilter(9))
                        #img0 = img0.filter(ImageFilter.EDGE_ENHANCE_MORE)
                        #img0 = img0.filter(ImageFilter.MaxFilter(3))
                        #img0 = img0.filter(ImageFilter.MinFilter(3))
                        img0 = cv2.cvtColor(np.array(img0),cv2.COLOR_RGB2BGR) #PIL.Image to OpenCV
                        
                        
                        gray1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #轉灰階
                        sobelx = cv2.Sobel(gray1, cv2.CV_64F, 1, 0)
                        sobely = cv2.Sobel(gray1, cv2.CV_64F, 0, 1)
                        sobelx = np.uint8(np.absolute(sobelx))
                        sobely = np.uint8(np.absolute(sobely))
                        sobelcombine1 = cv2.bitwise_or(sobelx,sobely)        
                        canny1 = cv2.Canny(gray1, 60, 140)
                        canny1 = np.uint8(np.absolute(canny1))
                        ret1, binary1 = cv2.threshold(gray1, 60, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
                        contours, hierarchy = cv2.findContours(gray1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        #print(contours)
                        draw_img3 = cv2.drawContours(CheckArea.copy(), contours, -1, (0, 255, 0), 1)
                        
                        total_area=(gray1.shape[0]*gray1.shape[1])*255
                        #print(Area*255)
                        #print(total_area)
                        #print(((Area*255)/total_area))
                        
                        cv2.putText(gray1,'Value:'+str(round((np.sum(gray1)/total_area)*100,4))+'% / '+str(Set_VALUE1[i])+'%',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
                        cv2.putText(sobelcombine1,'Value:'+str(round((np.sum(sobelcombine1)/total_area)*100,4))+'%',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
                        cv2.putText(canny1,'Value:'+str(round((np.sum(canny1)/total_area)*100,4))+'%',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
                        
                        Value1 = round((np.sum(gray1)/total_area)*100,4)      
                        
                        cv2.namedWindow('Particle Detection Area Compare with initial image'+str(j), cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Particle Detection Area Compare with initial image'+str(j), 600, 200)
                        cv2.imshow("Particle Detection Area Compare with initial image"+str(j), np.hstack([gray1,sobelcombine1,canny1])) #show drawing        
                        cv2.namedWindow('Particle Detection contours image'+str(j), cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Particle Detection contours image'+str(j), 200, 200)
                        cv2.imshow("Particle Detection contours image"+str(j), np.hstack([draw_img3])) #show drawing
        
                
                        filename = CheckArea
                        Draw_img,Value2 = amhs4_lmguide_value2.get_value2(filename)
                        
                        cv2.putText(Draw_img,'Value2:'+str(round(Value2,4))+'% / '+str(Set_VALUE2[i])+'%',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
                        cv2.imshow("Detection Value2"+str(j), np.hstack([Draw_img])) #show drawing
                        
                        if Value1>Set_VALUE1[i]:
                            print("Value 1 :"+str(Value1)+" over setting level :"+str(Set_VALUE1[i]))
                            Alarm_Flag=True
                        if Value2>Set_VALUE2[i]:
                            print("Value 2 :"+str(Value2)+" over setting level :"+str(Set_VALUE2[i]))
                            Alarm_Flag=True
                            
                        
                        filepath = "./logfile1.csv"
                        # 檢查檔案是否存在
                        if os.path.isfile(filepath):
                          print("Log file exist.")
                          fileprint=open(filepath, "a")
                        else:
                          print("Log file not exist.")
                          fileprint=open(filepath, "a")
                          fileprint.write("DEVICE_ID,DEVICE_NUMBER,FILENAME,POSITION,VALUE1,VALUE2\n")
                          
                        fileprint.write(str(DEVICE_ID[i])+","+str(DEVICE_NUMBER[i])+","+str(file)+","+str(j)+","+str(Value1)+","+str(Value2)+"\n")
                        
                        time.sleep(Delay_Time) #delay a little bit, if require
                        fileprint.close()                       
                    
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            
            #Check output folder and make dirs
            outfolderpath = "./OutputImage/"+str(DEVICE_ID[i])+"_"+str(DEVICE_NUMBER[i]) 
            if os.path.isdir(outfolderpath):
                print(str(DEVICE_ID[i])+"_"+str(DEVICE_NUMBER[i])+" Output folder exist.")                
            else:
                print(str(DEVICE_ID[i])+"_"+str(DEVICE_NUMBER[i])+" Output folder not exist.")
                os.makedirs(outfolderpath)
    
            if not os.path.isdir(outfolderpath+"/Alarm"):
                os.makedirs(outfolderpath+"/Alarm")
            if not os.path.isdir(outfolderpath+"/Normal"):
                os.makedirs(outfolderpath+"/Normal")
                
            if Alarm_Flag==True:
                print(str(DEVICE_ID[i])+"_"+str(DEVICE_NUMBER[i])+" Alarm_Flag is True.")
                shutil.move(folderpath+"/"+file, outfolderpath+"/Alarm")
                Alarm_Flag=False
            else:
                print(str(DEVICE_ID[i])+"_"+str(DEVICE_NUMBER[i])+" Alarm_Flag is False.")
                shutil.move(folderpath+"/"+file, outfolderpath+"/Normal")
            
            
        
        else:
                print(str(DEVICE_ID[i])+"_"+str(DEVICE_NUMBER[i])+" Input folder not exist.")                
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
    
    