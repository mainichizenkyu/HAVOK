 # -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:59:18 2018

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
from scipy import signal

b=8/3
def functionlorenz(x):
    return np.array((10*x[1]-10*x[0]+0*np.sqrt(0.01)*np.random.randn(),x[0]*28-x[0]*x[2]-x[1],x[0]*x[1]-b*x[2]),dtype=float)


def functionex1(x,a,t):
    return np.array((x[1],-0.3*x[0]-0.4*x[1]+0*(np.sin(2*np.pi*1*t)-1/2*np.sin(4*np.pi*1*t)+1/3*np.sin(6*np.pi*1*t)-1/4*np.sin(8*np.pi*1*t)+1/5*np.sin(10*np.pi*1*t)))+np.sqrt(0.001)*np.random.randn(),dtype=float)

def fft(x,t,n):
    x_fft=sp.fftpack.fft(x)
    #fftfreq, x_psd = signal.welch(x, fs = 100)
    x_psd=np.abs(x_fft)**2
    fftfreq = sp.fftpack.fftfreq(len(x_psd),1./1000)
    i=fftfreq>0
    plt.figure(figsize=(10,12))
    plt.subplot(211)
    plt.plot(t[0:100000],x[0:100000])
    plt.xlabel('Time(s)')
    n=n+1
    plt.ylabel('V'+str(n))
    plt.subplot(212)
    plt.plot(fftfreq[i],20*np.log10(x_psd[i]))
    plt.xlim(0,30)
    #plt.ylim(-30,90)
    plt.xlabel('Frequency(1/s)')
    plt.ylabel('PSD(dB)')
h=0.001
r=6
#ローレンツ方程式からデータの取得
rungekutta=RungeKutta1(functionlorenz)
x,y=rungekutta.calculation([10,10,10],h,200000)
plt.plot(x[:,0])
"""x=np.delete(x,np.arange(0,10000),0)
y=np.delete(y,np.arange(0,10000),0)"""
cal=Havok(x[:,0])
havok1,A1,B1,v1,dv1,vdf,V,S,U=cal.calculation2(100,r,r,h)
plt.rcParams["font.size"] = 30
t=np.linspace(0,199.9,199900)
plt.figure(figsize=(16,6))
plt.plot(t,V[0,:],label="$V_1$")
plt.plot(t[2:],havok1[:,0],label="havok")
plt.xlabel("dimensionless time")
plt.xlim(0, 50)
plt.grid()
plt.legend(loc="lower right",fontsize=15)


#外部入力成分の閾値による色分け
t=np.linspace(0,199.9,199900)
ratio = 0.9
vrborder=np.sort(np.abs(V[r,:]))[int(len(V[r, :])*0.85)]
vrsqrt=[]
inter_vr=[]
none_vr = []
t1_vr=[]
t2_vr=[]

for i in range(len(V[0,:])-100):
    vrsqrt.append(np.sqrt(np.var(V[r,i:i+100])))
for i in range(len(V[0,:])-100):
    if vrsqrt[i]>vrborder and vrsqrt[i-1]<vrborder:
        inter_temp=[]
        t1_temp=[]
        time = i
        while vrsqrt[time]>vrborder:
            inter_temp.append(-1*V[0,time+50])
            t1_temp.append(t[time+50])
            time += 1
            if time > 190000:
                break
        inter_vr.append(inter_temp)
        t1_vr.append(t1_temp)
    elif vrsqrt[i]<vrborder and vrsqrt[i-1] >vrborder:
        none_temp=[]
        t2_none=[]
        time = i
        while vrsqrt[time]<vrborder:
            none_temp.append(-1*V[0,time+50])
            t2_none.append(t[time+50])
            time += 1
            if time > 190000:
                break
        none_vr.append(none_temp)
        t2_vr.append(t2_none)
    else:
        pass
plt.rcParams["font.size"]=40
fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(16,12))
ax2.plot(t,V[r,:])
for i in range(len(none_vr)):
    if i < len(none_vr)-1:
        ax1.plot(t2_vr[i],none_vr[i], color = '#377eb8')
    else:
        ax1.plot(t2_vr[i],none_vr[i], color = '#377eb8', label = 'forcing inactive')
for i in range(len(inter_vr)):
    if i < len(inter_vr)-1:
        ax1.plot(t1_vr[i],inter_vr[i], color = '#ff7f00')
    else:
        ax1.plot(t1_vr[i],inter_vr[i], color = '#ff7f00',label = 'forcing active')
ax1.set_ylabel("$v_1(t)$")
ax2.set_xlabel("Dimensionless time")
ax2.set_ylabel("$v_6(t)$")
ax2.set_xlim(10,60)
ax1.legend(loc="lower right",fontsize=15)
fig.text(0, 0.90, '(a)', ha = 'left')
fig.text(0, 0.45, '(b)', ha = 'left')
fig.align_labels([ax1, ax2])
ax1.grid()
ax2.grid()
plt.show()






plt.figure(figsize=(18,9))
plt.plot(t[0:175000],V[0,0:175000],label="v1")
plt.plot(t[0:175000],havok1[0:175000,0],label="havok")
plt.xlabel("time(s)")
plt.ylabel("$v_1$")
plt.legend()

for i in range(r):
    fft(V[i,:],t,i)
    

plt.figure(figsize=(15,9))
for i in range(10):
    plt.plot(U[:,i],label='r='+str(i))
plt.xlabel("t")
plt.ylabel("$u_r$")
plt.legend()
    


dc=[]
for i in range(V.shape[1]-2):
    dc.extend(np.transpose((V[:,i+1]-V[:,i])**2))
dc=np.reshape(dc,((V.shape[1])-2,-1))
print((sum(dc[:,0]))**0.5)
dx=[x[:,0]]
dy=[x[:,1]]
dz=[x[:,2]]
for i in range(11):
    dx.append(10*(dy[i]-dx[i]))
    com=[]
    for j in range(i+1):
        com.append(comb(i,j,exact=True)*dx[j]*dz[i-j])
    com=np.array(com).T
    com=np.sum(com,axis=1)
    print(com)
    dy.append(28*dx[i]-dy[i]-com)
    com1=[]
    for j in range(i+1):
        com1.append(comb(i,j,exact=True)*dx[j]*dy[i-j])
    com1=np.array(com1).T
    com1=np.sum(com1,axis=1)
    dz.append(com1-b*dz[i])
    

plt.rcParams["font.size"] = 20
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 12))
ax1.plot(x[:, 0])
ax2.plot(dx[5])
ax1.set_ylabel("$x^{(0)}(t)$", fontsize=40)
ax2.set_ylabel("$x^{(10)}(t)$", fontsize=40)
ax2.set_xlabel("Step", fontsize=40)
ax2.set_xlim(4000, 24000)
ax1.grid()
ax2.grid()
plt.show()

#線形微分方程式の解の成分の表示
expAt=lambda s :sp.linalg.expm(-1*A1*s)
s=np.linspace(0,10,10000)
expAt115=[]
for i in range(10000):
    expAs=expAt(s[i])
    expAt115.append(expAs[0,r-2])
plt.figure(figsize=(12,6))
plt.plot(s,expAt115)
plt.xlabel("time")
plt.ylabel("$exp(-At)_{1,14}$")

#高階微分の閾値ともとの時系列データの関係性の図示
t=np.linspace(0,200.001,200001)
t=np.linspace(0,200.001,200001)
Vborder=np.sort(np.abs(dx[r-1]))[int(len(dx[r-1])*0.85)]
x9=dx[r-1]
inter=[]
none=[]
t1=[]
t2=[]
interval=600
ran=0.001
x9sqrt=[]
for i in range(int(x9.size-100)):
    x9sqrt.append(np.sqrt(np.var(x9[i:i+100])))
for i in range(int(x9.size)-100):
    if x9sqrt[i]>Vborder and x9sqrt[i-1]<Vborder:
        inter_temp=[]
        t1_temp=[]
        time = i
        while x9sqrt[time]>Vborder:
            inter_temp.append(x[time+50,0])
            t1_temp.append(t[time+50])
            time += 1
            if time > 190000:
                break
        inter.append(inter_temp)
        t1.append(t1_temp)
    elif x9sqrt[i]<Vborder and x9sqrt[i-1] >Vborder:
        none_temp=[]
        t2_none=[]
        time = i
        while x9sqrt[time]<Vborder:
            none_temp.append(x[time+50, 0])
            t2_none.append(t[time+50])
            time += 1
            if time > 190000:
                break
        none.append(none_temp)
        t2.append(t2_none)
    else:
        pass
plt.rcParams["font.size"]=40
fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(16,12))
ax2.plot(t,x9)
for i in range(len(none)):
    if i < len(none)-1:
        ax1.plot(t2[i],none[i], color = '#377eb8')
    else:
        ax1.plot(t2[i], none[i], color = '#377eb8', label='precursor inactive')
for i in range(len(inter)):
    if i < len(inter)-1:
        ax1.plot(t1[i],inter[i], color = '#ff7f00')
    else:
        ax1.plot(t2[i], inter[i], color = '#ff7f00', label = 'precursor active')
ax1.set_ylabel("$x(t)$")
ax1.set_ylim(-20, 20)
ax2.set_xlabel("Dimensionless time")
ax2.set_ylabel("$x^{(5)}(t)$")
ax2.set_xlim(10,60)
ax1.legend(loc="lower right",fontsize=15)
fig.text(0, 0.90, '(a)', ha = 'left')
fig.text(0, 0.45, '(b)', ha = 'left')
fig.align_labels([ax1, ax2])
ax1.grid()
ax2.grid()
plt.show()



#相空間上での高階微分の変動の記述

t=np.linspace(0,200.001,200001)
t=np.linspace(0,200.001,200001)
Vborder=5*10**6
x9=dx[r-1]
interx=[]
intery=[]
interz=[]
nonex=[]
noney=[]
nonez=[]
t1=[]
t2=[]
interval=600
ran=0.001
x9sqrt=[]
for i in range(int(x9.size-100)):
    x9sqrt.append(np.sqrt(np.var(x9[i:i+100])))
for i in range(int(x9.size)-100):
    if x9sqrt[i]>Vborder and x9sqrt[i-1]<Vborder:
        interx_temp=[]
        intery_temp=[]
        interz_temp=[]
        t1_temp=[]
        time = i
        while x9sqrt[time]>Vborder:
            interx_temp.append(x[time+50, 0])
            intery_temp.append(x[time+50, 1])
            interz_temp.append(x[time+50, 2])
            t1_temp.append(t[time+50])
            time += 1
            if time > 190000:
                break
        interx.append(interx_temp)
        intery.append(intery_temp)
        interz.append(interz_temp)
        t1.append(t1_temp)
    elif x9sqrt[i]<Vborder and x9sqrt[i-1] >Vborder:
        nonex_temp=[]
        noney_temp=[]
        nonez_temp=[]
        t2_none=[]
        time = i
        while x9sqrt[time]<Vborder:
            nonex_temp.append(x[time+50, 0])
            noney_temp.append(x[time+50, 1])
            nonez_temp.append(x[time+50, 2])
            t2_none.append(t[time+50])
            time += 1
            if time > 190000:
                break
        nonex.append(nonex_temp)
        noney.append(noney_temp)
        nonez.append(nonez_temp)
        t2.append(t2_none)
    else:
        pass
plt.rcParams["font.size"]=30
plt.rcParams["axes.labelpad"]=20
fig=plt.figure(figsize=(10,8))
ax=Axes3D(fig)
for i in range(len(interx)):
    if i < len(interx)-1: 
        ax.plot(interx[i],intery[i],interz[i], color = '#377eb8')
    else:
        ax.plot(interx[i],intery[i],interz[i], color = '#377eb8', label = 'forcing active')
for i in range(len(nonex)):
    if i < len(nonex)-1:
        ax.plot(nonex[i],noney[i], nonez[i], color = '#ff7f00')
    else:
        ax.plot(nonex[i],noney[i], nonez[i], color = '#ff7f00', label='forcing inactive')
ax.set_xlabel("$x(t)$")
ax.set_ylabel("$y(t)$")
ax.set_zlabel("$z(t)$")
ax.legend(loc="lower right",fontsize=15)
ax.grid()
plt.show()

def functionlinear(A,x):
    return np.dot(A,x)

n=50000
rungekutta2=RungeKutta1(functionlinear)
X=rungekutta2.calculation1(havok1[0,:],h,A1,n)


plt.rcParams["axes.labelpad"]=23
fig=plt.figure(figsize=(10,8))
ax=Axes3D(fig)
ax.plot(X[:,0],X[:,1],X[:,2],".",ms=1.3)
ax.set_xlabel("$V_1$   ",fontsize=18,linespacing=100)
ax.set_ylabel("   $V_2$",fontsize=18,linespacing=50)
ax.set_zlabel("   $V_3$",fontsize=18,linespacing=50)
ax.zaxis.set_tick_params(pad=15)
plt.show()
plt.rcParams["axes.labelpad"]=10
plt.rcParams["font.size"] = 11

vr_distri, vr_x = np.histogram(V[8, :], density = True, bins = 41)
X_vr = []
for i in range(len(vr_distri)):
    X_vr.append((vr_x[i]+vr_x[i+1])/2)
dx8_distri, dx8_x = np.histogram(dx[8]/np.std(dx[8])/np.sqrt(200000), density = True, bins = 41)

X_dx8 = []
for i in range(len(dx8_distri)):
    X_dx8.append((dx8_x[i]+dx8_x[i+1])/2)  
plt.rcParams["font.size"] = 35  
plt.figure(figsize = (12, 8))
plt.yscale("log")
plt.xlim(-0.025, 0.025)
plt.plot(X_vr, vr_distri, label = '$v_9$')
plt.plot(X_dx8, dx8_distri, label = '$x^{(8)}$')
plt.legend(fontsize = 25)
plt.xlabel('$v_9, x^{(8)}$')
plt.ylabel('probability density')



