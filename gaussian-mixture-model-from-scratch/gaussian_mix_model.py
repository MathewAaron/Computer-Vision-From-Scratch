# -*- coding: utf-8 -*-
"""

@author: Aaron
unity id: asmathew

"""
"""
This is the function to compute Gaussian Mixture Model

psuedo code:
    
    1. Training data: Computing mean, covariance
    2. Computing MLE (i.e EM Algorithm)
    3. Computing posterior probability
    4. Plotting ROC curve

References: Lec04- Stage 1 ppt

"""


from numpy.linalg import inv, det
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve,auc
import cv2
from tqdm import tqdm
# Defining K as 1
K = 3
# Defining PCA dimensions
pca_dims = 100

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
       
trface, trnface, tsface , tsnface = load_pickle_data()

# Converting 2D array to 1D
trface = trface.reshape(trface.shape[0],-1) # Storing all training faces
trnface = trnface.reshape(trnface.shape[0],-1) # Storing all training background images
tsface = tsface.reshape(tsface.shape[0],-1) # Storing all testing images
tsnface = tsnface.reshape(tsnface.shape[0],-1) # Storing all testing background images

trface,pca_f =  reduce_array_dim(trface)
trnface,pca_nf = reduce_array_dim(trnface)
tsface,_ = reduce_array_dim(tsface)
tsnface,_ = reduce_array_dim(tsnface)

trface,trnface,tsface,tsnface = trface.T,trnface.T,tsface.T,tsnface.T
    
class GaussianMixModel():
    
    """
    Gaussian mixture model using EM Algorithm
    """
    def __init__ (self,mean_matrix,covar_matrix,size):
        
        self.mean_matrix = mean_matrix
        self.covar_matrix = covar_matrix
        self.size = size
        self.length = self.mean_matrix.shape[1]
        self.weight = np.random.dirichlet(np.ones(K),size = 1)[0]
        self.post_prob = np.random.dirichlet(np.ones(K),size = self.size)
        
    def compute_EM(self,count,i,data_array):
        
        # Compute the E step
        
        compute_arr = 0
        
        for j in range(0,K):
            compute_arr = np.add(compute_arr, np.multiply(self.weight[j],self.logpdf(j,count,data_array)))
        
        self.post_prob[count,i] = np.divide(np.multiply(self.weight[i] , self.logpdf(i,count,data_array)),compute_arr)
        
        # Computing weights and maximizing it

        num1 = 0
        den = 0
        for count in range(self.size):
            
            num1 = np.add(num1,self.post_prob[count,i])
            
            for j in range(K):
                
                den = np.add(den , self.post_prob[count,j])
        
        self.weight[i] = np.divide(np.multiply(1.0,num1),den)
        
        np.zeros((self.length,1))
        den = 0
        
        for count in range(0,self.size):
            
            num1 = num1 + self.post_prob[count,i] * np.matmul((data_array[:,count].reshape(-1,1) - self.mean_matrix[i]) ,(data_array[:,i].reshape(-1,1) - self.mean_matrix[i]).T)
            
            den = np.add(den,self.post_prob[count,i])
            
        self.covar_matrix[i] = np.divide(np.multiply(1.0,num1),den)
        self.covar_matrix[i] = np.diag(np.diag(self.covar_matrix[i]))
        
    def fit_model(self,i,data_array):
        for j in range(self.size):
            self.compute_EM(j,i ,data_array)
    
    def logpdf(self,count,i,data_array):
        
        pdf_var = np.matmul((data_array[:,i].reshape(-1,1) - self.mean_matrix[count]).T, inv(self.covar_matrix[count]))
        
        pdf_var2 =  -0.5 * np.matmul(pdf_var,data_array[:,i].reshape(-1,1) - self.mean_matrix[count])
        
        log_pdf = np.exp(pdf_var2) / (np.sqrt(det(self.covar_matrix[k]) * (2*np.pi ** data_array.shape[0])))
            
        return log_pdf
    
    def compute_probability(self,count,data_array):
        
        prob_variable = 0
        
        for i in range(K):
            prob_variable = np.multiply(np.add(prob_variable, self.weight[i]), self.logpdf(i, count, data_array))
        
        return prob_variable
     

if __name__ == '__main__':
    
    mean_arr_f = np.zeros((K,100,1))
    
    covar_arr_f = np.array([np.random.uniform(low=0.0, high=1.0, size=(pca_dims,pca_dims))* np.identity(100) for k in range(K)])
    gauss_model_face = GaussianMixModel(mean_arr_f ,covar_arr_f, 1000)
    
    #initializing the gmm model for nonface data
    mean_arr_nf = np.zeros((K,100,1))
    covar_arr_nf = np.array([np.random.uniform(low=0.0, high=1.0, size=(pca_dims,pca_dims)) * np.identity(100) for k in range(K)])
    gauss_model_nface = GaussianMixModel(mean_arr_nf ,covar_arr_nf, 1000)
    
    
    # Computing Gaussian Mix model for face
    for k in tqdm(range(K),desc="Progress"):
        gauss_model_face.fit_model(k,trface)
        
    
    # Computing Gaussian Mix model for non-face
    for k in tqdm(range(K),desc="Progress"):    
        gauss_model_nface.fit_model(k,trnface)    
        
    
    prob_f_face = np.array([])
    prob_nf_face = np.array([])
    prob_f_nface = np.array([])
    prob_nf_nface = np.array([])
    
    #Running predictions on test data
    for i in range(test_size):
        prob_f_face = np.append( prob_f_face , gauss_model_face.compute_probability(i,tsface) )
        prob_f_nface =np.append(prob_f_nface , gauss_model_face.compute_probability(i,tsnface) )
        prob_nf_face =  np.append(prob_nf_face , gauss_model_nface.compute_probability(i,tsface) )
        prob_nf_nface =np.append(prob_nf_nface , gauss_model_nface.compute_probability(i,tsnface) )
        
    #calculating the posterior probabilities        
    PP_F_F = prob_f_face / (prob_f_face + prob_nf_face)
    PP_NF_F = prob_nf_face / (prob_f_face + prob_nf_face)
    PP_F_NF = prob_f_nface / (prob_f_nface + prob_nf_nface)
    PP_NF_F = prob_nf_nface / (prob_f_nface + prob_nf_nface)
    
    compute_posterior(prob_f_face,prob_nf_face,prob_f_nface,prob_nf_nface)
    ROC_curve(np.nan_to_num(PP_F_NF),np.nan_to_num(PP_F_F),test_size)
    
    