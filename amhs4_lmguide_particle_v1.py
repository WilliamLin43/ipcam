# -*- coding: utf-8 -*-
import cv2 #load opencv
import numpy as np
from numpy import genfromtxt
from PIL import Image, ImageEnhance, ImageFilter

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
    
    filename = cv2.imread("./defect.png")
    img=filename
       
    #read points record from file
    dataPath=r'./cutpoints0.csv'
    points1=genfromtxt(dataPath,delimiter=',').astype(int).tolist()
    
    #cut image & save log
    CheckArea1=croppoly(img,points1)
    cv2.imwrite("./cut1.png", CheckArea1)
    cv2.imshow("CheckArea1", np.hstack([CheckArea1])) #show drawing
    
    #read points record from file
    dataPath=r'./cutpoints1.csv'
    points2=genfromtxt(dataPath,delimiter=',').astype(int).tolist()
    #cut image & save log
    CheckArea2=croppoly(img,points2)
    cv2.imwrite("./cut1.png", CheckArea2)
    cv2.imshow("CheckArea2", np.hstack([CheckArea2])) #show drawing
    
    
    
    pts1=np.asarray(points1)
    pts1 = pts1.reshape((-1,1,2)) 
    pts2=np.asarray(points2)
    pts2 = pts2.reshape((-1,1,2))
    img2=cv2.polylines(img, [pts1], True, (0, 0, 255), 4)
    img2=cv2.polylines(img, [pts2], True, (0, 0, 255), 4)
    
    cv2.namedWindow('Particle Detection Image & Area', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Particle Detection Image & Area', 1000, 500)
    cv2.imshow("Particle Detection Image & Area", np.hstack([img2])) #show drawing
    
    img0 = Image.fromarray(cv2.cvtColor(CheckArea1,cv2.COLOR_BGR2RGB)) #OpenCV to PIL.Image
    img0 = img0.filter(ImageFilter.ModeFilter(3))
    #img0 = img0.filter(ImageFilter.EDGE_ENHANCE_MORE)
    #img0 = img0.filter(ImageFilter.MaxFilter(3))
    #img0 = img0.filter(ImageFilter.MinFilter(3))
    img0 = cv2.cvtColor(np.array(img0),cv2.COLOR_RGB2BGR) #PIL.Image to OpenCV
    
    
    gray1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #轉灰階
    
    #30 and 150 is the threshold, larger than 150 is considered as edge,
    #less than 30 is considered as not edge
    canny1 = cv2.Canny(gray1, 100, 170)
    canny1 = np.uint8(np.absolute(canny1))
    ret, binary1 = cv2.threshold(gray1, 100, 150, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    cv2.namedWindow('Particle Detection Area 1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Particle Detection Area 1', 800, 200)
    cv2.imshow("Particle Detection Area 1", np.hstack([gray1,canny1,binary1])) #show drawing
    
    
    img0 = Image.fromarray(cv2.cvtColor(CheckArea2,cv2.COLOR_BGR2RGB)) #OpenCV to PIL.Image
    img0 = img0.filter(ImageFilter.ModeFilter(3))
    img0 = img0.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img0 = img0.filter(ImageFilter.MaxFilter(3))
    #img0 = img0.filter(ImageFilter.MinFilter(3))
    img0 = cv2.cvtColor(np.array(img0),cv2.COLOR_RGB2BGR) #PIL.Image to OpenCV
    
    gray2 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #轉灰階
    
    #30 and 150 is the threshold, larger than 150 is considered as edge,
    #less than 30 is considered as not edge
    canny2 = cv2.Canny(gray2, 100, 170)
    canny2 = np.uint8(np.absolute(canny2))
    ret, binary2 = cv2.threshold(gray2, 100, 150, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    
    cv2.namedWindow('Particle Detection Area 2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Particle Detection Area 2', 800, 200)
    cv2.imshow("Particle Detection Area 2", np.hstack([gray2,canny2,binary2])) #show drawing
    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
    
    