# -*- coding: utf-8 -*-
"""

@author: Aaron
unity id: asmathew
"""
import cv2
import numpy as np
import sys
from os import path
import math
import pickle
from data_traning import *
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt
from numpy.linalg import inv,det
import sys
sys.path.append("D:/NCSU Subjects/NCSU Sem 2/ECE 763/project01/")
from utils.loadData import *
from utils.data_traning import *

train_faces,train_non_faces,test_faces,test_non_faces = load_pickle_data()

    
train_faces = np.array(train_faces)
test_faces= np.array(test_faces)
train_non_faces= np.array(train_non_faces)
test_non_faces= np.array(test_non_faces)

    
class Gaussian():
    
    def __init__(self):
        self.object1 = load_pickle_data()
        
    def pdf(self,array_mean,i,mean,cov_matrix):
        
        x = array_mean[:,i] - mean
        num_term = -0.5*np.matmul((np.matmul(x.T,inv(cov_matrix))),x)
        den_term =np.sqrt(np.power(2*np.pi,len(mean))*det(cov_matrix))
        
        return 1.*np.exp(num_term)/den_term
    
    def compute_gauss_model(self,face_train_pca,nface_train_pca):
        
        mean_face = np.mean(face_train_pca,axis=1)#face_train_pca.mean(axis=0)
        mean_nface = np.mean(nface_train_pca,axis=1) #nface_train_pca.mean(axis=0)
        cv1 = np.cov(face_train_pca,rowvar= True)
        
        cov_face = np.diag(np.diag(cv1))
        cov_nface = np.diag(np.diag(np.cov(nface_train_pca,rowvar= True)))
        
        return mean_face,mean_nface,cov_face,cov_nface
    


if __name__ == "__main__" :
    
    facetr,nfacetr,facets,nfacets = load_pickle_data()
    #facetr = np.nan_to_num(facetr)
    ftrain, pca_val_f = reduce_array_dim(facetr)
    nftrain, pca_val_nf = reduce_array_dim(nfacetr)
    ftest, _ = reduce_array_dim(facets)
    nftest, _ = reduce_array_dim(nfacets)
    
    ftrain,nftrain,ftest,nftest = ftrain.T,nftrain.T,ftest.T,nftest.T
    model = Gaussian()
    mean_f,mean_nf,var_f,var_nf=  model.compute_gauss_model(ftrain,nftrain)
    
    prob_f_face = np.array([])
    prob_nf_face = np.array([])
    prob_f_nface = np.array([])
    prob_nf_nface = np.array([])
    
    for i in range(test_size):
        prob_f_face = np.append(prob_f_face,model.pdf(ftest,i,mean_f,var_f))
        prob_f_nface = np.append(prob_f_nface,model.pdf(nftest,i,mean_f,var_f))
        prob_nf_face = np.append(prob_nf_face,model.pdf(ftest,i,mean_nf,var_nf))
        prob_nf_nface = np.append(prob_nf_nface,model.pdf(nftest,i,mean_nf,var_nf))
        
    #calculating the posterior probabilities        
    PP_F_F = prob_f_face / (prob_f_face + prob_nf_face)
    PP_NF_F = prob_nf_face / (prob_f_face + prob_nf_face)
    PP_F_NF = prob_f_nface / (prob_f_nface + prob_nf_nface)
    PP_NF_F = prob_nf_nface / (prob_f_nface + prob_nf_nface)
    
    compute_posterior(prob_f_face,prob_nf_face,prob_f_nface,prob_nf_nface)
    ROC_curve(PP_F_NF,PP_F_F ,test_size)
    
    
    








