# -*- coding: utf-8 -*-
"""
@author: Aaron Mathew
unity id: asmathew

"""

import numpy as np
from numpy.linalg import inv,det
from scipy.special import gamma,digamma,gammaln
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
from tqdm import tqdm



# Initializing estimated matrix
def compute_cost(len_arr,est_matrix1,est_matrix2):
     
    length_mat = len(est_matrix1)
    coeff_term_1 = (len_arr/2) * np.log((len_arr/2))
    coeff_term_2 = gammaln((len_arr/2))
    total = 0
    for i in range(length_mat):
        coeff_term_3 = ((len_arr/2) - 1) * est_matrix2[i]
        coeff_term_4 = (len_arr/2) * est_matrix1[i]
        total = total + coeff_term_1 - coeff_term_2 + coeff_term_3 - coeff_term_4
    total = -total
    return total

# Function to minimize the scalar value

class ComputeTDistribution():
    """
    This code contains a class for T-distribution model
    
    psuedo code for computing t-distribution
        
        1. Load Images
        2. Compute PDF
            i. reduce dimension of data/ flatten data
            ii. Compute mean and covariance of the data
        3. Apply EM algorithm
            i. Using digamma function to comupte log-derivative [reference from ppt (lec 5)]
            ii. Optimize it to give minimization of scalar functions
            iii. Compute cost of the function
        4. Compute posterior probability
        5. Plot ROC
    
"""
    def __init__(self,mean_matrix,covar_matrix,len_arr):
        self.mean_matrix = mean_matrix
        self.covar_matrix = covar_matrix
        self.len_arr = len_arr
        self.est_matrix1 = np.zeros(train_size)
        self.est_matrix2 = np.zeros(train_size)
        self.d_sigma = np.zeros(train_size)
        
    def compute_EM(self,data_array):
        
        shape_coeff = self.mean_matrix.shape[0]
        
        for i in range(0,train_size):
            var_coeff = np.matmul( (data_array[:,i].reshape(-1,1)-self.mean_matrix).T , inv(self.covar_matrix) )
            d_sigma = np.matmul(var_coeff , (data_array[:,i].reshape(-1,1) - self.mean_matrix))                                  
            self.d_sigma[i] = d_sigma
            self.est_matrix1[i] = (self.len_arr+shape_coeff)/(self.len_arr + d_sigma)
            self.est_matrix2[i] = digamma((self.len_arr+shape_coeff)/2) - np.log((self.len_arr+d_sigma)/2)
                
        #mean update
        self.mean_matrix = (np.sum(self.est_matrix1 * data_array, axis=1)/np.sum(self.est_matrix1)).reshape(shape_coeff,1)
    
        #covariance update
        num = np.zeros((shape_coeff,shape_coeff))
        for i in range(0,train_size):
            prod = np.matmul((data_array[:,i].reshape(-1,1) - self.mean_matrix), (data_array[:,i].reshape(-1,1) - self.mean_matrix).T)
            num = num + self.est_matrix1[i]*prod
        self.covar_matrix = num/np.sum(self.est_matrix1)
        self.covar_matrix = np.diag( np.diag(self.covar_matrix) )
        
        #updating dof via argmin
        self.len_arr = self.optimize_scalar_min()
     
        for i in range(0,train_size):
            var_coeff = np.matmul( (data_array[:,i].reshape(-1,1)-self.mean_matrix).T , inv(self.covar_matrix) )                                  
            self.d_sigma[i] = np.matmul(var_coeff , (data_array[:,i].reshape(-1,1) - self.mean_matrix))
    def optimize_scalar_min(self):
        len_var = fminbound(compute_cost, 0, 10, args=(self.est_matrix1, self.est_matrix2))
        
        return len_arr
    
    def compute_probability(self,idx,data_array):
        sh_array = self.mean_matrix.shape[0]
        val1 = gamma((self.len_arr + sh_array)/2) / ( ((self.len_arr * np.pi)** sh_array/2) *np.sqrt(det(self.covar_matrix))*gamma(self.len_arr/2) )
        var_coeff = np.matmul( (data_array[:,idx].reshape(-1,1)-self.mean_matrix).T,inv(self.covar_matrix) )                                  
        d_sigma = np.matmul(var_coeff,(data_array[:,idx].reshape(-1,1) - self.mean_matrix))
        val2 = (1 + d_sigma/self.len_arr)
        val = val1 * pow(val2, -(self.len_arr+sh_array)/2)
        return val[0,0]
    def optimize_scalar_min(self):
        len_arr = fminbound(compute_cost, 0, 10, args=(self.est_matrix1, self.est_matrix2))
        
        return len_arr
    

