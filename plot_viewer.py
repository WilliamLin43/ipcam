# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # 動圖的核心函式
import seaborn as sns  # 美化圖形的一個繪圖包

sns.set_style("whitegrid")  # 設定圖形主圖

# 建立畫布
fig, ax = plt.subplots()
fig.set_tight_layout(True)

fp = pd.read_csv("ipclist.csv")#read data
DEVICE_ID = fp.DEVICE_ID #Device ID
DEVICE_NUMBER = fp.DEVICE_NUMBER #Device Number
IP = fp.IP #IP address data
PORT = fp.PORT #Port data
USER = fp.USER #User data
PWD = fp.PWD #Password data
Set_VALUE1=fp.VALUE1
Set_VALUE2=fp.VALUE2



fp2 = pd.read_csv("logfile.csv")#read data
Data_DEVICE_ID = fp2.DEVICE_ID #Device ID
Data_DEVICE_NUMBER = fp2.DEVICE_NUMBER #Device Number
VALUE1 = fp2.VALUE1
VALUE2 = fp2.VALUE2
VALUE1newData=np.array(VALUE1).transpose() #行列互換
VALUE2newData=np.array(VALUE2).transpose() #行列互換
#print(len(data))
print(np.char.count(str(Data_DEVICE_ID),str(fp.DEVICE_ID[0])))
DEVICE_ID_Count_Value = np.zeros(shape=(3,np.sum(np.char.count(str(Data_DEVICE_ID),str(fp.DEVICE_ID[0])))))
print(DEVICE_ID_Count_Value)

n=0
for k in range(len(VALUE1newData)):
    if fp.DEVICE_ID[0]==fp2.DEVICE_ID[k]:        
        DEVICE_ID_Count_Value[0,n]=fp2.DEVICE_NUMBER[k]
        DEVICE_ID_Count_Value[1,n]=fp2.VALUE1[k]
        DEVICE_ID_Count_Value[2,n]=fp2.VALUE2[k]
        n=n+1
        
print(DEVICE_ID_Count_Value)
        
    
# 畫出一個維持不變（不會被重畫）的散點圖和一開始的那條直線。
VALUE1newDatax = np.arange(0, len(VALUE1newData),1)
VALUE1newDatay = VALUE1newData
VALUE2newDatay = VALUE2newData
#print(x)
#print(y)
cValue = ['r','y','g','b','r','y','g','b','r'] 
#ax.scatter(x,y,c=cValue,marker='s') 
ax.scatter(VALUE1newDatax, VALUE1newDatay,c=cValue[0])
ax.scatter(VALUE1newDatax, VALUE2newDatay,c=cValue[1])

VALUE1newDatay_Line = np.zeros(len(VALUE1newDatax))
VALUE1newDatay_avge=0
VALUE2newDatay_Line = np.zeros(len(VALUE1newDatax))
VALUE2newDatay_avge=0

for j in range(len(VALUE1newDatax)):
    VALUE1newDatay_avge=(VALUE1newDatay_avge+VALUE1newDatay[j])/(j+1)
    VALUE2newDatay_avge=(VALUE2newDatay_avge+VALUE2newDatay[j])/(j+1)
    #print(VALUE1newDatay_avge)
    VALUE1newDatay_Line[j] = VALUE1newDatay_avge
    VALUE2newDatay_Line[j] = VALUE2newDatay_avge

line, = ax.plot(VALUE1newDatax, VALUE1newDatay_Line, c=cValue[0], linewidth=2)
line, = ax.plot(VALUE1newDatax, VALUE2newDatay_Line, c=cValue[1], linewidth=2)

def update(i):
    label = 'timestep {0}'.format(i)
    #print(label)
    # 更新直線和x軸（用一個新的x軸的標籤）。
    # 用元組（Tuple）的形式返回在這一幀要被重新繪圖的物體
    #line.set_ydata(x - 5 + i)  # 這裡是重點，更新y軸的資料
    ax.set_xlabel(label)    # 這裡是重點，更新x軸的標籤
    return line, ax

# FuncAnimation 會在每一幀都呼叫“update” 函式。
# 在這裡設定一個10幀的動畫，每幀之間間隔200毫秒
anim = FuncAnimation(fig, update, frames=np.arange(0, len(IP)), interval=200)

plt.show()

  
    
    
