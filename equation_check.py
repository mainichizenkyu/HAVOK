# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:19:20 2021

@author: ryuch
"""


import numpy as np
import matplotlib.pyplot as plt
from havok import Havok
from rungekutta import RungeKutta1, RungeKutta3
import scipy as sp
import scipy.fftpack
from scipy.special import perm, comb
from mpl_toolkits.mplot3d import Axes3D

#ローレンツ方程式の定義
#ローレンツ方程式の定義
b=8/3
sigma = 10
gamma = 28

#各種パラメータの定義
N = 200000
m = 100
N_ = N-m+1


plt.rcParams["font.size"] = 30

def functionlorenz(x):
    return np.array((sigma*x[1]-sigma*x[0],x[0]*gamma-x[0]*x[2]-x[1],x[0]*x[1]-b*x[2]),dtype=float)    
    
def calculate_higher_order_derivative_Lorenz(x, N):
    dx=[x[:,0]]
    dy=[x[:,1]]
    dz=[x[:,2]]
    for i in range(N):
        dx.append(10*(dy[i]-dx[i]))
        com=[]
        for j in range(i+1):
            com.append(comb(i,j,exact=True)*dx[j]*dz[i-j])
        com=np.array(com).T
        com=np.sum(com,axis=1)
        dy.append(28*dx[i]-dy[i]-com)
        com1=[]
        for j in range(i+1):
            com1.append(comb(i,j,exact=True)*dx[j]*dy[i-j])
        com1=np.array(com1).T
        com1=np.sum(com1,axis=1)
        dz.append(com1-b*dz[i])
    return dx, dy, dz
    
def calculate_kappa(x, N):
    return np.sum(x*x)/N

def calculate_coefficient_for_r_6(dx):
    kappa = []
    for i in range(len(dx)):
        kappa.append(calculate_kappa(dx[i], N_))
    s0 = np.sqrt(kappa[0])
    s1 = np.sqrt(kappa[1])
    s2 = np.sqrt(kappa[2]-kappa[1]**2/kappa[0])
    s3 = np.sqrt(kappa[3]-kappa[2]**2/kappa[1])
    s4 = np.sqrt(kappa[4]-kappa[2]**2/kappa[0]-(kappa[1]*kappa[2]-kappa[0]*kappa[3])**2/kappa[0]/(kappa[0]*kappa[2]-kappa[1]**2))
    s5 = np.sqrt(kappa[5]-kappa[3]**2/kappa[1]-(kappa[2]*kappa[3]-kappa[1]*kappa[4])**2/kappa[1]/(kappa[1]*kappa[3]-kappa[2]**2))
    print(s0, s1, s2, s3, s4, s5)
    print(s1/s0, s2/s1, s3/s2, s4/s3, s5/s4)
    a = [[0,s1/s0,0,0,0],[-s1/s0,0,s2/s1,0,0],[0,-s2/s1,0,s3/s2,0],[0,0,-s3/s2,0,s4/s3],[0,0,0,-s4/s3,0]]
    b = [0,0,0,0,s5/s4]
    return [s0, s1, s2, s3, s4, s5], kappa,np.array(a),np.array(b)

def construct_initialization(dx,s,k):
    init = []
    init.append(1/s[0]*dx[0][0])
    init.append(1/s[1]*dx[1][0])
    init.append(1/s[2]*(dx[2][0]+dx[0][0]*k[1]/k[0]))
    init.append(1/s[3]*(dx[3][0]+dx[1][0]*k[2]/k[1]))
    init.append(1/s[4]*(dx[4][0]+dx[2][0]*(k[1]*k[2]-k[0]*k[3])/(k[1]**2-k[0]*k[2])+dx[0][0]*(k[2]**2-k[1]*k[3])/(k[1]**2-k[0]*k[2])))
    #init.append(1/s[5]*(dx[5][0]+dx[3][0]*(k[2]*k[3]-k[1]*k[4])/(k[2]**2-k[1]*k[3])+dx[1][0]*(k[3]**2-k[2]*k[4])/(k[2]**2-k[1]*k[3])))
    return np.array(init)

def construct_input(dx,s, k):
    f = 1/s[5]*(dx[5]+dx[3]*(k[2]*k[3]-k[1]*k[4])/(k[2]**2-k[1]*k[3])+dx[1]*(k[3]**2-k[2]*k[4])/(k[2]**2-k[1]*k[3]))
    return f


def calculate1(x,h,v15,v152,B,A,m):
    s1=np.dot(A,x)+np.reshape(v15*B,(m-1))
    s2=np.dot(A,x+0.5*s1*h)+np.reshape((v15+v152)*0.5*B,(m-1))
    s3=np.dot(A,x+0.5*s2*h)+np.reshape((v15+v152)*0.5*B,(m-1))
    s4=np.dot(A,x+h*s3)+np.reshape(v152*B,(m-1))
    return x+h/6*(s1+2*s2+2*s3+s4)
    
def solve_linear_eq(A, B, v0, v6, h = 0.001, r = 6):
    res = [v0]
    for i in range(len(v6)-1):
        v1= calculate1(v0,h,v6[i],v6[i+1],B,A, r)
        res.append(v1)
        v0=v1
    return res
    
    
if __name__ == "__main__":
    
    
    #ローレンツ方程式からデータの取得
    h=0.001
    rungekutta=RungeKutta1(functionlorenz)
    x,y=rungekutta.calculation([10,10,10],h,N)
    plt.plot(x[:,0])

    #HAVOKによる結果の取得
    r=6
    cal=Havok(x[:,0])
    havok1,A1,B1,v1,dv1,vdf,V,S,U=cal.calculation2(m,r,r,h)


    #周波数分析
    t=np.linspace(0,199.900,199900) 

    #高階微分の算出 
    dx, dy, dz = calculate_higher_order_derivative_Lorenz(x, r-1)
    s, k, a, b = calculate_coefficient_for_r_6(dx)
    
    dx = np.array(dx)
    dx = dx[:, 1:]
    
    v0 = construct_initialization(dx.tolist(), s, k)
    v6 = construct_input(dx,s, k)
    
    plt.figure(figsize = (12, 6))
    plt.plot(v6/np.sqrt(N_), label = "input_theory")
    plt.plot(V[5, :], label = "input_svd")
    plt.legend()
    plt.xlabel("time step")
    
    res = np.array(solve_linear_eq(a, b, v0, v6, h, r))
    plt.figure(figsize = (12, 6))
    plt.plot(res[:, 0], label = "result_theory")
    plt.plot(dx[0]/s[0], label = "v1_theory")
    plt.legend()
    plt.xlabel("time step")
    
    t=np.linspace(0,200.001,200001) 