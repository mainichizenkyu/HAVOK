# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:30:02 2019

@author: amanuma_yuta
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

#周波数解析の関数の定義
def fft_v(x,t,n):
    plt.rcParams["font.size"] = 30
    x_fft=sp.fftpack.fft(x)
    x_psd=np.abs(x_fft)**2
    fftfreq = sp.fftpack.fftfreq(len(x_psd),1./1000)
    i=fftfreq>0
    plt.figure(figsize=(12,6))
    plt.plot(fftfreq[i],10*np.log10(x_psd[i]))
    plt.xlim(0,30)
    plt.grid()
    plt.xlabel('Frequency')
    plt.ylabel('PSD(dB) of $v_{%d}$' % n)
    
def make_figure_fft_v(x, t, num):
    plt.rcParams["font.size"] = 30
    
    fig, ax = plt.subplots(len(num),1,sharex=True,figsize=(12,6*len(num)))
    for fig_i, n in enumerate(num):
        x_fft=sp.fftpack.fft(x[n])
        x_psd=np.abs(x_fft)**2
        fftfreq = sp.fftpack.fftfreq(len(x_psd),1./1000)
        i=fftfreq>0
        ax[fig_i].plot(fftfreq[i],10*np.log10(x_psd[i]))
        ax[fig_i].set_xlim(0,30)
        ax[fig_i].set_ylim(-50,30)
        ax[fig_i].grid()
        ax[fig_i].set_ylabel('PSD(dB) of $v_{%d}$' % (n+1))
    plt.xlabel('Frequency')
    #base = 1/len(num)
    #for chr_s in reversed(range(len(num))):
        #print(1-(base-0.05)*chr_s)
        #fig.text(0, 1-(base-0.05)*chr_s, '(%c)' % chr(97+chr_s), ha = 'left')
    fig.text(0, 0.88, '(a)', ha = 'left')
    fig.text(0, 0.6, '(b)', ha = 'left')
    fig.text(0, 0.33, '(c)', ha = 'left')
    
def fft_x(x,t,n):
    plt.rcParams["font.size"] = 30
    x_fft=sp.fftpack.fft(x)
    x_psd=np.abs(x_fft)**2
    fftfreq = sp.fftpack.fftfreq(len(x_psd),1./1000)
    i=fftfreq>0
    plt.figure(figsize=(12,6))
    plt.plot(fftfreq[i],10*np.log10(x_psd[i]))
    plt.xlim(0,30)
    plt.grid()
    plt.xlabel('Frequency')
    plt.ylabel('PSD(dB) of $x^{(%d)}$' % n)
    
def make_figure_fft_x(x, t, num):
    plt.rcParams["font.size"] = 30
    
    fig, ax = plt.subplots(len(num),1,sharex=True,figsize=(12,6*len(num)))
    for fig_i, n in enumerate(num):
        x_fft=sp.fftpack.fft(x[n]/np.std(x[n])/np.sqrt(N_))
        x_psd=np.abs(x_fft)**2
        fftfreq = sp.fftpack.fftfreq(len(x_psd),1./1000)
        i=fftfreq>0
        ax[fig_i].plot(fftfreq[i],10*np.log10(x_psd[i]))
        ax[fig_i].set_xlim(0,30)
        ax[fig_i].set_ylim(-50,30)
        ax[fig_i].grid()
        ax[fig_i].set_ylabel('PSD(dB) of $x^{(%d)}$' % (n))
    plt.xlabel('Frequency')
    #base = 1/len(num)
    #for chr_s in reversed(range(len(num))):
        #print(1-(base-0.05)*chr_s)
        #fig.text(0, 1-(base-0.05)*chr_s, '(%c)' % chr(97+chr_s), ha = 'left')
    fig.text(0, 0.88, '(a)', ha = 'left')
    fig.text(0, 0.6, '(b)', ha = 'left')
    fig.text(0, 0.33, '(c)', ha = 'left')
    

    
    
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
    return [s0, s1, s2, s3, s4, s5], kappa
    
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

    for i in range(r):
        fft_v(V[i, :], t, i+1)


    #高階微分の算出 
    dx, dy, dz = calculate_higher_order_derivative_Lorenz(x, r-1)
    s, k = calculate_coefficient_for_r_6(dx)
    
    
    num = [1, 3, 5]
    make_figure_fft_v(V, t, num)
    make_figure_fft_x(dx, t, num)
    
    t=np.linspace(0,200.001,200001)    

    