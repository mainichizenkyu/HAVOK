# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:50:10 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Havok:
     def __init__(self,X):
         #Xは横に状態変数縦に時系列を並べた行列
         self.x=X
     def hankelx(self,q,DF):
         #ハンケル行列の計算、qは列にとる変数の数
         H=[]
         for i in range(q):
             H.append(self.x[i:len(self.x)-1-q+i])
         H=np.reshape(H,(q,-1))
         U, S, V = np.linalg.svd(H,full_matrices=False) 
         v15=V[DF-1,:]
         plt.rcParams["axes.labelpad"]=20
         fig=plt.figure(figsize=(10,8))
         ax=Axes3D(fig)
         ax.plot(V[0,:],V[1,:],V[2,:],".",ms=1.3)
         ax.set_xlabel("$V_1$   ",fontsize=18,linespacing=100)
         ax.set_ylabel("   $V_2$",fontsize=18,linespacing=50)
         ax.set_zlabel("   $V_3$",fontsize=18,linespacing=50)
         ax.zaxis.set_tick_params(pad=15)
         plt.show()
         plt.rcParams["axes.labelpad"]=10
         plt.rcParams["font.size"] = 11

         return H,V,v15,S,U
    
     def derivation(self,v,h):
         #データVの時間微分を計算する
         dv=[]
         for i in range(v.shape[1]-2):
             dv.extend(np.transpose((v[:,i+2]-v[:,i])/(2*h)))
         dv=np.reshape(dv,((v.shape[1])-2,-1))
         return dv
     
     def derivation5(self,v,h):
         #データVの時間微分を計算する
         dv=[]
         for i in range(v.shape[1]-5):
             dv.extend(np.transpose((v[:,i]-8*v[:,i+1]+8*v[:,i+3]-v[:,i+4])/(12*h)))
         dv=np.reshape(dv,((v.shape[1])-5,-1))
         return dv
     
     def SINDy(self,v,dv):
         #⓵lstsqで近似⓶閾値より小さい成分を0⓷0でない成分について再度各変数ごとでlstsq
         A,_,_,_=np.linalg.lstsq(v,dv)
         for k in range(10):
             smallinds=(np.abs(A)<0.1)
             A[smallinds]= 0
             for ind in range(dv.shape[1]):
                 biginds = (smallinds==False)[:,ind]
                 A[biginds,ind],_,_,_=np.linalg.lstsq(v[:,biginds], dv[:,ind])
             A=np.transpose(A)
         return A
     
     def calculate1(self,x,h,v15,v152,B,A,m):
         s1=np.dot(A,x)+np.reshape(v15*B,(m-1))
         s2=np.dot(A,x+0.5*s1*h)+np.reshape((v15+v152)*0.5*B,(m-1))
         s3=np.dot(A,x+0.5*s2*h)+np.reshape((v15+v152)*0.5*B,(m-1))
         s4=np.dot(A,x+h*s3)+np.reshape(v152*B,(m-1))
         return x+h/6*(s1+2*s2+2*s3+s4)
     
     def calculation1(self,q,drivingF,m):
         #3点近似による計算
         #qはHavokの縦に取る変数の個数
         #mは方程式を近似する際の次元,基本的にはmと同じ値でよい
         #vdfのdfはdriving forceの意味、外力とみる成分をいくつとするかを定める
         H,v,vdf= self.hankelx(q,drivingF)
         v=np.delete(v,range(m,v.shape[0]),0)
         dv=self.derivation(v)
         v=np.delete(v,0,1)
         v=np.delete(v,range(30000,v.shape[1]),1)
         v=np.transpose(v)
         dv=np.delete(dv,range(30000,dv.shape[0]),0)
         A=self.SINDy(v,dv)
         A1=np.delete(A,m-1,0)
         A1=np.delete(A1,m-1,1)

         B=np.delete(A,np.arange(0,m-1),1)
         B=np.delete(B,m-1,0)
         x0=v[0:14,0]
         print(x0.shape)
         havok=[x0]
         for i in range(len(vdf)):
             x1=self.calculate1(x0,0.001,vdf[i+1],B,A1)
             havok.append(x1)
             x0 =x1
         print(x0.shape)
         #havokは横に状態変数、縦に時系列を格納している
         return np.array(havok),A1,B,v,dv
     
     def calculation2(self,q,drivingF,m,h):
         #微分を5点近似で求める
         #qはHavokの縦に取る変数の個数
         #mは方程式を近似する際の次元,基本的にはmと同じ値でよい
         #vdfのdfはdriving forceの意味、外力とみる成分をいくつとするかを定める
         H,v1,vdf,S,U= self.hankelx(q,drivingF)
         vdf=np.delete(vdf,[0,1],0)
         v=np.delete(v1,range(m,v1.shape[0]),0)
         dv=self.derivation5(v,h)
         v=np.delete(v,[0,1],1)
         v=np.delete(v,range(30000,v.shape[1]),1)
         v=np.transpose(v)
         dv=np.delete(dv,range(30000,dv.shape[0]),0)
         print(np.shape(v))
         print(np.shape(dv))
         A=self.SINDy(v,dv)
         A1=np.delete(A,m-1,0)
         A1=np.delete(A1,m-1,1)

         B=np.delete(A,np.arange(0,m-1),1)
         B=np.delete(B,m-1,0)
         x0=v[0,0:m-1]
         print(x0.shape)
         havok=[x0]
         for i in range(len(vdf)-1):
             x1=self.calculate1(x0,h,vdf[i],vdf[i+1],B,A1,m)
             havok.append(x1)
             x0 =x1
         #havokは横に状態変数、縦に時系列を格納している
         return np.array(havok),A1,B,v,dv,vdf,v1,S,U
