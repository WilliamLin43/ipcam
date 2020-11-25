# PyQt5與OpenCV的簡單集成
import sys
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np

class WinForm(QMainWindow):
    def __init__(self, parent=None):
        super(WinForm, self) .__init__(parent)  
        self.setGeometry(400, 150, 1138, 799)   #視窗起始位置，視窗大小
        layout = QVBoxLayout ()
        self.btn1 = QPushButton ('開啟圖片', self)
        self.btn1.setGeometry(10, 10, 60, 30)   #按鈕起始位置，按鈕大小
        self.btn1.clicked.connect (self.open)     #連接
       
        self.btn2 = QPushButton ('關閉視窗', self)
        self.btn2.setGeometry(770, 10, 60, 30)
        self.btn2.clicked.connect (self.close)
        
        self.btn3 = QPushButton ('存檔', self)
        self.btn3.setGeometry(700, 10, 60, 30)
        self.btn3.clicked.connect (self.save)
        
        self.btn4 = QPushButton ('HSV', self)
        self.btn4.setGeometry(80, 10, 60, 30)
        self.btn4.clicked.connect (self.HSV)
        
        self.btn5 = QPushButton ('模糊', self)
        self.btn5.setGeometry(150, 10, 60, 30)
        self.btn5.clicked.connect (self.blurre)
        
        self.btn6 = QPushButton ('灰階', self)
        self.btn6.setGeometry(220, 10, 60, 30)
        self.btn6.clicked.connect (self.gray)
        
        self.btn7 = QPushButton ('二值化', self)
        self.btn7.setGeometry(290, 10, 60, 30)
        self.btn7.clicked.connect (self.canny)
        
        self.btn8 = QPushButton ('膨脹', self)
        self.btn8.setGeometry(360, 10, 60, 30)
        self.btn8.clicked.connect (self.dilated)
        
        self.btn9 = QPushButton ('侵蝕', self)
        self.btn9.setGeometry(430, 10, 60, 30)
        self.btn9.clicked.connect (self.eroded)
        
        self.btn10 = QPushButton ('高倍偵測', self)
        self.btn10.setGeometry(500, 10, 60, 30)
        self.btn10.clicked.connect (self.circle)
        
        self.btn11 = QPushButton ('低倍偵測', self)
        self.btn11.setGeometry(570, 10, 60, 30)
        self.btn11.clicked.connect (self.circle1)
        
        self.label = QLabel("", self)
        self.label.setGeometry(10, 45, 554, 739)
        layout.addWidget (self.label)
        self.label2 = QLabel("", self)
        self.label2.setGeometry(574, 45, 554, 739)
        layout.addWidget (self.label2)
        self.setLayout (layout)
        self.setWindowTitle ('PyQt5測試')
        
    def open(self):    #開啟圖片
        # File, _ = QFileDialog.getOpenFileName (self,  'Open File', 'c:\\', "Image Files (*.jpg *.jpeg *.png)")
        File, _  = QFileDialog.getOpenFileName(self, 'Open Image', './__data', '*.png *.jpg *.bmp')
        if File is '':
            return
        self.img = cv2.imread(File, -1)
        if self.img.size == 1:#意思为图像的通道数为1，通道数为1表示是灰度图
            return
        self.Show()
        self.refreshShow()
      
    def blurre(self):   #模糊
        if self.img.size == 1:
            return  
        self.img = cv2.GaussianBlur(self.img, (9, 9), 0)  #高斯和的寬與高
        self.refreshShow()
        
    def HSV(self):      #HSV偵測
        if self.img.size == 1:
            return     
        lower_green = np.array([30,0,0])
        upper_green = np.array([150,255,255])    
        mask = cv2.inRange(self.img, lower_green, upper_green)
        self.img = cv2.bitwise_and(self.img, self.img, mask = mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))  #定義元素結構(以3x3 5x5 7x7 來設定)
        self.img = cv2.dilate(self.img,kernel,5) #膨脹(數字為次數)
        self.img = cv2.erode(self.img,kernel,5) #侵蝕
        self.refreshShow()
       
    def gray(self):    #灰階
        if self.img.size == 1:
            return     
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.GRAYrefreshShow()
        
    def canny(self):   #二值化
        if self.img.size == 1:
            return     
        self.img = cv2.Canny(self.img,50, 25) #圖片,閾值1,閾值2
        self.GRAYrefreshShow()
        
    def dilated(self):   #膨脹
        if self.img.size == 1:
            return     
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  
        #定義元素結構(以3x3 5x5 7x7 來設定)
        self.img = cv2.dilate(self.img,kernel,1) #膨脹(數字為次數)
        self.GRAYrefreshShow()  
        
    def eroded(self):   #侵蝕
        if self.img.size == 1:
            return     
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  
        #定義元素結構(以3x3 5x5 7x7 來設定)
        self.img = cv2.erode(self.img,kernel,1) #侵蝕
        self.GRAYrefreshShow()
        
    def circle(self):      #高倍偵測
        if self.img.size == 1:
            return
        circle = cv2.HoughCircles(self.img,cv2.HOUGH_GRADIENT,1,15,param1=50,param2=20,minRadius=5,maxRadius=30)   
        # 影像,檢測方法,累加器圖像,圓心最小距離,最高閾值(默認100,此值一半為低閾值),
        # 偵測圓形(默認100,數值越大越接近完美的圓),最小圓半徑,最大圓半徑
        i=0
        if circle is not None:
            for c in circle:
                for x,y,r in c:
                    cv2.line(self.img,(x+r+r+r, y+r+r),(x+r,y+r), (255,255,255), 2)  
                    cv2.line(self.img,(x+r, y+r),(x+r+r,y+r+r+r), (255,255,255), 2)
                    cv2.line(self.img,(x+r+r, y+r+r+r),(x+r+r+r,y+r+r), (255,255,255), 2)
                    if (x>0 and y>0):
                        i+=1
                        print('總數:',i) 
        cv2.putText(self.img, 'Quantity:', (10,60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3, 
                    cv2.LINE_AA)  
        #影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類 
        cv2.putText(self.img, str(i), (330,60), cv2.FONT_HERSHEY_COMPLEX,
                    2, (255, 255, 255), 3, cv2.LINE_AA)
        self.GRAYrefreshShow()
        
    def circle1(self):      #低倍偵測
        if self.img.size == 1:
            return

        circle = cv2.HoughCircles(self.img,cv2.HOUGH_GRADIENT,1,5,
                                  param1=50,param2=13,minRadius=2,
                                  maxRadius=20)   
        #影像,檢測方法,累加器圖像,圓心最小距離,最高閾值(默認100,此值一半為低閾值)
        #,偵測圓形(默認100,數值越大越接近完美的圓),最小圓半徑,最大圓半徑
        i=0
        if circle is not None:
            for c in circle:
                for x,y,r in c:
                    cv2.line(self.img,(x+r+r+r, y+r+r),(x+r,y+r), (255,255,255), 2)  
                    cv2.line(self.img,(x+r, y+r),(x+r+r,y+r+r+r), (255,255,255), 2)
                    cv2.line(self.img,(x+r+r, y+r+r+r),(x+r+r+r,y+r+r),(255,255,255), 2)
                    if (x>0 and y>0):
                        i+=1
                        print('總數:',i) 
        cv2.putText(self.img, 'Quantity:', (10,60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3, 
                    cv2.LINE_AA)  
        #影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類 
        cv2.putText(self.img, str(i), (330,60), cv2.FONT_HERSHEY_COMPLEX,
                    2, (255, 255, 255), 3, cv2.LINE_AA)
        self.GRAYrefreshShow()
        
        
    def save(self):      #存檔
        FileSave, _ = QFileDialog.getSaveFileName (self,  'Save file', 'c:\\', "Image Files (*.jpg *.jpeg *.png)")   
        cv2.imwrite(FileSave, self.img)
       
       
    def Show(self):  
        # 提取图像的尺寸和通道, 用于将opencv下的image转换成Qimage
        height, width, channel = self.img.shape
        bytesPerLine = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine,
                           QImage.Format_RGB888).rgbSwapped()
        # 将Qimage显示出来
        self.label.setPixmap (QPixmap(self.qImg))
        self.label.setScaledContents (True)
        
    def refreshShow(self):
        # 提取图像的尺寸和通道, 用于将opencv下的image转换成Qimage
        height, width, channel = self.img.shape
        bytesPerLine = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine,
                           QImage.Format_RGB888).rgbSwapped()
        # 将Qimage显示出来
        self.label2.setPixmap (QPixmap(self.qImg))
        self.label2.setScaledContents (True)
        
    def GRAYrefreshShow(self):
        height, width = self.img.shape
        self.qImg = QImage(self.img.data, width, height,width, 
                            QImage.Format_Grayscale8)
        self.label2.setPixmap(QPixmap.fromImage(self.qImg))
        self.label2.setScaledContents (True)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = WinForm()
    win.show()
    sys.exit(app.exec_())