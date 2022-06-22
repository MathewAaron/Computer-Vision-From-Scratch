# -*- coding: utf-8 -*-
"""

@author: Aaron
"""
import cv2
import numpy as np
import math
import pickle
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from numpy.linalg import inv,det

def load_pickle_data():
    """
    Returns
    -------
    trface_arr : ndarray
        Array with all training images of Face Data
    trnface_arr : ndarray
        Array with all training images of Non-Face Data
    tsface_arr : ndarray
        Array with all testing images of Face Data
    tsnface_arr : ndarray
        Array with all testing images of Non-Face Data

    """
    trface_arr = np.zeros([1000,20,20])
    trnface_arr = np.zeros([1000,20,20])
    
    tsface_arr = np.zeros([1000,20,20])
    tsnface_arr = np.zeros([1000,20,20])
    
    face_test_path = "INSERT PATH FOR ALL TRAIN IMAGES"
    nonface_test_path = "INSERT PATH FOR ALL TEST IMAGES"
    for i in range(0,1000):
        trface = cv2.imread(face_test_path+'image_'+str(i)+'.jpg',0)
        trface = np.asarray(trface)
        trface_arr[i,:] = trface
        trnface = cv2.imread(nonface_test_path+'image_'+str(i)+'.jpg',0)
        trnface = trnface
        trnface_arr[i,:] = trnface
        
    for i in range(0,100):
        tsface = cv2.imread(face_test_path+'image_'+str(i)+'.jpg',0)
        tsface = np.asarray(tsface)
        tsface_arr[i,:] = tsface
        tsnface = cv2.imread(nonface_test_path+'image_'+str(i)+'.jpg',0)
        tsnface = tsnface
        tsnface_arr[i,:] = tsnface
    
    return trface_arr, trnface_arr, tsface_arr, tsnface_arr

def reduce_array_dim(carray):
    """
    
    Parameters
    ----------
    carray : ndarray
        Image data (Gray scale images only)

    Returns
    -------
    final_data : ndarray
        Flattened Images.
    pca : int
        PCA dimension used for flattening.

    """
    img_flatten_list = []
    pca = PCA(2)
    
    carray = np.nan_to_num(carray)
    pca.fit(carray)
    
    pca_transform = pca.transform(carray)
    temp = StandardScaler()
    temp.fit(pca_transform)
    final_data = temp.transform(pca_transform)
    
    return final_data, pca

def compute_posterior(pp_face_f,pp_nonface_f,pp_face_nf,pp_nonface_nf):
    """
    Function to compute posterior probabilities

    """
    compute_post_face_f = pp_face_f/(pp_face_f+pp_face_nf)
    compute_post_nface_f = pp_nonface_f/(pp_nonface_f+pp_face_f)
    compute_post_face_nf = pp_face_nf/(pp_nonface_f+pp_nonface_nf)
    compute_post_nface_nf = pp_nonface_nf/(pp_nonface_nf+pp_nonface_f)
    
    pos_face = np.sum(compute_post_face_f >= 0.5)
    pos_nonface = np.sum(compute_post_nface_nf >= 0.5)
    neg_face = 100 - pos_face
    neg_nonface = 100 - pos_nonface
    
    FPR = neg_nonface / (neg_nonface + pos_nonface)
    FNR = neg_face  / (pos_face + neg_face )
    MCR = (neg_nonface + neg_face) / 200
    
    print('False Positive Rate:' + str(FPR))
    print('False Negative Rate:' + str(FNR))
    print('Miss Classification Rate:' + str(MCR)) 
    
    return compute_post_face_f, compute_post_nface_f,compute_post_face_nf,compute_post_nface_nf

class Gaussian():
    """
    This class is used to compute a Gaussian Model for face detection
    The dataset used is FDDB dataset with images samples at 20x20 pixels.
    
    """
    def __init__(self):
        pass
        
    def pdf(self,array_mean,i,mean,cov_matrix):
        """
        Parameters
        ----------
        array_mean : ndarray
            A numpy array with Face/Non-Face Image data
        i : int
            Number of current image
        mean : float
            Mean of Face/ Nonface
        cov_matrix : ndarray
            Covariance matrix computed for Face/ Nonface

        Returns
        -------
        TYPE
            float

        """
        x = array_mean[:,i] - mean
        num_term = -0.5*np.matmul((np.matmul(x.T,inv(cov_matrix))),x)
        den_term =np.sqrt(np.power(2*np.pi,len(mean))*det(cov_matrix))
        
        return 1.*np.exp(num_term)/den_term
    
    def compute_gauss_model(self,face_train_pca,nface_train_pca):
        """

        Parameters
        ----------
        face_train_pca : ndarray
            Flattened Array for Face
        nface_train_pca : ndarray
            Flattened Array for Non-Face

        Returns
        -------
        mean_face : float
            Mean of face Images
        mean_nface : float
            Mean for non-face Images
        cov_face : ndarray
            Covariance for face Images
        cov_nface : ndarray
            Covariance Matrix for non face Images

        """
        mean_face = np.mean(face_train_pca,axis=1)#face_train_pca.mean(axis=0)
        mean_nface = np.mean(nface_train_pca,axis=1) #nface_train_pca.mean(axis=0)
        cv1 = np.cov(face_train_pca,rowvar= True)
        
        cov_face = np.diag(np.diag(cv1))
        cov_nface = np.diag(np.diag(np.cov(nface_train_pca,rowvar= True)))
        
        return mean_face,mean_nface,cov_face,cov_nface
    

if __name__ == "__main__" :
    
    test_size = 200
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
    
    
    








