# -*- coding: utf-8 -*-
import cv2
import numpy as np


filename = cv2.imread("./cut0.png")
img1=filename
'''
overlapping = cv2.addWeighted(filename1, 0.1, filename2, 0.1, 0)
frameDelta = cv2.absdiff(filename1, filename2)

(B,G,R) = cv2.split(filename2) # split G B R image

img0 = Image.fromarray(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)) #OpenCV to PIL.Image
im_invert = ImageOps.invert(img0)
im_invert = cv2.cvtColor(np.array(im_invert),cv2.COLOR_RGB2BGR) #PIL.Image to OpenCV
'''

lower_green = np.array([30,0,0])
upper_green = np.array([150,255,255])    
mask = cv2.inRange(img1, lower_green, upper_green)
img1 = cv2.bitwise_and(img1, img1, mask = mask)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  #定義元素結構(以3x3 5x5 7x7 來設定)
img1 = cv2.dilate(img1,kernel,1) #膨脹(數字為次數)
#img1 = cv2.erode(img1,kernel,1) #侵蝕
cv2.imshow("Particle Detection Image", np.hstack([img1])) #show drawing
cv2.waitKey(0)                 #等待按下任何按鍵

img1= cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img1= cv2.Canny(img1,50, 25) #圖片,閾值1,閾值2
cv2.imshow("Particle Detection Image", np.hstack([img1])) #show drawing
cv2.waitKey(0)                 #等待按下任何按鍵

  


kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  
#定義元素結構(以3x3 5x5 7x7 來設定)
img1 = cv2.dilate(img1,kernel,1) #膨脹(數字為次數)
cv2.imshow("Particle Detection Image", np.hstack([img1])) #show drawing
cv2.waitKey(0)

#img1 = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY)[1]

mask = np.zeros(filename.shape, np.uint8)
#mask=np.ones_like(mask,np.uint8)*0 #fill the rest with white
draw_img = mask
contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


for i in range(0,len(contours)):  
    x, y, w, h = cv2.boundingRect(contours[i])
    #print(w)
    #print(img1.shape[1])
    if (w<(img1.shape[1]*0.6)):
        if(h<(img1.shape[0]*0.6)):
            draw_img = cv2.drawContours(draw_img.copy(), contours, i, (0, 0, 255), 3)
        #cv2.waitKey(0)                 #等待按下任何按鍵
        #cv2.imwrite( input_dir+str(i)+".jpg",new_img)
        
total_area=(img1.shape[0]*img1.shape[1])*255

cv2.putText(draw_img,'Value:'+str(round((np.sum(draw_img)/total_area)*100,4))+'%',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
cv2.imshow("Particle Detection Image", np.hstack([draw_img])) #show drawing
cv2.waitKey(0)

         

'''
imgshape = img1.shape
avreage=0
count=0
     
for i in range(imgshape[0]):
    for j in range(imgshape[1]):
        if img1[i][j]>0:
            avreage=avreage+img1[i][j]
            count=count+1

avreage = avreage/count  #計算像素平均值
print(int(avreage))
Variance=np.var(img1)
print(int(Variance))
Standard_Deviation = np.std(img1,ddof=1)
print(int(Standard_Deviation))



#_,imgae1= cv2.threshold(img1, 255, 0, cv2.THRESH_BINARY)
_,imgae1= cv2.threshold(img1, int(avreage), 255, cv2.THRESH_BINARY)


cv2.imshow("Particle Detection Image", np.hstack([imgae1])) #show drawing

cv2.waitKey(0)                 #等待按下任何按鍵
'''
cv2.destroyAllWindows()