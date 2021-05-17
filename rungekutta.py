# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:45:35 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt


class RungeKutta1:
    def __init__(self, equation):
        self.f=equation

    def calculate(self,x,h):
        f=self.f
        s1=f(x)
        s2=f(x+0.5*s1*h)
        s3=f(x+0.5*s2*h)
        s4=f(x+h*s3)
        return x+h/6*(s1+2*s2+2*s3+s4),s4


    def calculation(self,x0,h,n):
        x=[x0]
        y=[]
        for i in range(n):
            x1,y1=self.calculate(x0,h)
            x.append(x1)
            y.append(y1)
            x0 =x1
        return np.array(x),np.array(y)
    
    def calculate1(self,x,h,A):
        f=self.f
        s1=f(A,x)
        s2=f(A,x+0.5*s1*h)
        s3=f(A,x+0.5*s2*h)
        s4=f(A,x+h*s3)
        return x+h/6*(s1+2*s2+2*s3+s4),s4

    
    def calculation1(self,x0,h,A,n):
        x=[x0]
        y=[]
        for i in range(n):
            x1,y1=self.calculate1(x0,h,A)
            x.append(x1)
            y.append(y1)
            x0 =x1
        return np.array(x)
    

    
class RungeKutta2:
    def __init__(self, equation):
        self.f=equation
    
    def calculate(self,x,h,a):
        f=self.f
        s1=f(x,a)
        s2=f(x+0.5*s1*h,a)
        s3=f(x+0.5*s2*h,a)
        s4=f(x+h*s3,a)
        return x+h/6*(s1+2*s2+2*s3+s4),h/6*(s1+2*s2+2*s3+s4)
    
    def calculation(self,x0,h,n,a):
        x=[x0]
        y=[]
        for i in range(n):
            x1,y1=self.calculate(x0,h,a)
            x.append(x1)
            y.append(y1)
            x0 = x1
        return np.array(x),np.array(y)
    
class RungeKutta3:
    def __init__(self, equation):
        self.f=equation
    
    def calculate(self,x,h,a,t):
        f=self.f
        s1=f(x,a,t)
        s2=f(x+0.5*s1*h,a,t)
        s3=f(x+0.5*s2*h,a,t)
        s4=f(x+h*s3,a,t)
        return x+h/6*(s1+2*s2+2*s3+s4),h/6*(s1+2*s2+2*s3+s4)
    
    def calculation(self,x0,h,n,a):
        x=[x0]
        y=[]
        t=np.linspace(0,h*n,n+1)
        print(t)
        for i in range(n):
            x1,y1=self.calculate(x0,h,a,t[i])
            x.append(x1)
            y.append(y1)
            x0 =x1
        return np.array(x),np.array(y),t
    
    
    