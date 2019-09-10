# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:16:34 2019

@author: Tan Phan

@Topic : Calculate some standard parameters of a data

PLAT CAT AND MOUSE WITH DESCR

"""
from sklearn.datasets import load_breast_cancer
import numpy as np
from scipy import stats

class Paras:
    def __init__(self,data):
        self._mean = 0
        self._mod = 0
        self._medium = 0
        self._sd = 0
        self._se = 0
        self._len = len(data)
    def check_DESCR(self, data, n):
        nparray_data = data[:,n]
        min_np = np.amin(nparray_data)
        max_np = np.amax(nparray_data)
        return [min_np, max_np]
    # Mean
    def mean_range(self,data, n):
        _mean = np.mean(data[:,n])
        return _mean
    #Mode
    def mode(self,data,n):        
        _mod = np.mod(data[:,n])
        return _mod
    #Medium
    def medium(self, data,n):
        _med = np.median(data[:,n])
        return _med
#    Standard Deviation
    def stand_dev(self, data,n):
        _sd = np.std(data[:,n])
        return _sd
        #Margin of Error
    #Standard Error
    def stand_error(self,data,n):
        _se = stats.sem(data[:,n])
        return _se

    
ori_dataset = load_breast_cancer()
dataset= ori_dataset.data
test = Paras(dataset)
a = test.mean_range(dataset)

# Find the mean every 
b = dataset[:,0]   
min_b = np.amin(b)
max_b = np.amax(b)

c = test.check_DESCR(dataset, 1)
    
    

