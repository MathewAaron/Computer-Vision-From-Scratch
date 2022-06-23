# -*- coding: utf-8 -*-
"""
@author: Aaron
"""

"""
    This file contains the class for adaboost using EM Algorithm
    
"""

from haar_feature import HaarFeatureExtraction
import numpy as np
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, haar_like_feature_coord,draw_haar_like_feature
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2

def load_pickle_data():
    trface_arr = np.zeros([500,20,20])
    trnface_arr = np.zeros([500,20,20])
    
    tsface_arr = np.zeros([100,20,20])
    tsnface_arr = np.zeros([100,20,20])
    
    face_test_path = "INSERT FACE DATASET PATH"
    nonface_test_path = "INSERT NON-FACE DATASET PATH"
    
    for i in range(0,500):
        trface = cv2.imread(face_test_path+'image_'+str(i)+'.jpg',0)
        trface = trface
        trface_arr[i,:] = np.asarray(trface)
        trnface = cv2.imread(nonface_test_path+'image_'+str(i)+'.jpg',0)
        trnface = trnface
        trnface_arr[i,:] = np.asarray(trnface)
    for i in range(0,100):
        tsface = cv2.imread(face_test_path+'image_'+str(i)+'.jpg',0)
        tsface = tsface
        tsface_arr[i,:] = np.asarray(tsface)
        tsnface = cv2.imread(nonface_test_path+'image_'+str(i)+'.jpg',0)
        tsnface = tsnface
        tsnface_arr[i,:] = np.asarray(tsnface)     
    return np.asarray(trface_arr,dtype =np.int64), np.asarray(trnface_arr,dtype =np.int64),\
                        np.asarray(tsface_arr,dtype =np.int64),np.asarray(tsnface_arr,dtype =np.int64)


class Adaboost(HaarFeatureExtraction):
    
    """
    This class computes Adaboost algorithm using the best weak classifier and EM algorithm
    """
    
    def __init__(self,combined_feature_list,threshold,len_dataset):
        
        self.weights = None
        self.alpha = []
        self.error = np.zeros(combined_feature_list.shape[1])
        self.haar_index = []
        self.Z = 0
        self.H = 0
        
    def get_weak_classifiers(self,combined_feature_list,threshold,len_dataset):
        
        thresh_val = combined_feature_list >= threshold 
    # get correct samples
        corr_samples = np.vstack((np.where(thresh_val[:len_dataset,:]==False,-1,1*thresh_val[:len_dataset,:]),
                         np.where(~thresh_val[len_dataset:,:]==False,1,-1*~thresh_val[len_dataset:,:])))
    # get incorrect samples
        incorr_samples = np.vstack((np.where(thresh_val[:len_dataset,:]==False,1,0*thresh_val[:len_dataset,:]),
                               np.where(~thresh_val[len_dataset:,:]==False,1,0*~thresh_val[len_dataset:,:])))
    
        return corr_samples, incorr_samples
    
    def adaboost_compute(self,combined_feature_list,threshold,len_dataset,num_epochs):
        
        self.weights =np.ones(len_dataset*2)/(len_dataset*2)
        face_data_weight = np.ones(len_dataset)
        nface_data_weight = np.copy(face_data_weight)*-1
        
        combined_weight_list = np.hstack((face_data_weight,nface_data_weight))
        
        corr_sample, incorr_sample = self.get_weak_classifiers(combined_feature_list, threshold, len_dataset)
        
        haar_arr = []
        
        for i in tqdm(range(num_epochs),desc = "{Computing Adaboost}"):
            
            self.error = np.dot(self.weights,incorr_sample)
            haar_temp = np.argsort(self.error)[0]
            print("Epoch {} : H(t) = {}".format(i+1,haar_temp))
            self.haar_index.append(haar_temp)
            
            haar_arr.append(combined_feature_list[:,haar_temp])
            
            error_temp = self.error[haar_temp]
            
            alpha_temp = (1/2) * np.log((1-error_temp)/(error_temp))
            self.alpha.append(alpha_temp)
            exp_temp = np.multiply(combined_weight_list,corr_sample[:,haar_temp]) * -1 * alpha_temp
            self.Z = np.multiply(self.weights,np.exp(exp_temp)).sum()
            self.weights = np.multiply(self.weights,np.exp(exp_temp)) / self.Z
            self.H = np.sign(np.dot(self.alpha,haar_arr))
            error_comp = (np.sum(self.H != combined_weight_list))
            
            print("Error: {} , Alpha: {}, Z: {}".format(error_comp,alpha_temp,self.Z))
            
        
        return self.haar_index, self.alpha
    
    
    def get_adaboost_haar(self,img,index_,f_type,f_coord):
        int_img = integral_image(np.asarray(img))
        return haar_like_feature(int_img, 0, 0, int_img.shape[0],int_img.shape[1],\
                      f_type[index_:index_+1],f_coord[index_:index_+1])
    
# get adaboost given best haar features list    
    def get_adaboost_haar_f(self,index,dataset,feature_type):
        f_coord,f_type = haar_like_feature_coord(dataset[0].shape[0],dataset[0].shape[1],feature_type)
        H_list = [np.asarray([self.get_adaboost_haar(data,ind,f_type,f_coord)[0] for data in dataset])\
              for ind in tqdm(index)]
        return np.asarray(np.nan_to_num(H_list))
 
    # Predict the model by providing testing data
    def predict(self,combined_test_data,haar_index_t,alpha_t,feature):
        
        num_data = np.int(len(combined_test_data)/2)
        
        ts_face_data = combined_test_data[:100]
        ts_nface_data = combined_test_data[100:]
        ts_face_data_weight = np.ones(num_data)
        ts_nface_data_weight = np.ones(num_data) * -1
        
        combined_face_data_weight = np.hstack((ts_face_data_weight,ts_nface_data_weight))
        haar_feature_test_data = self.get_adaboost_haar_f(haar_index_t,combined_test_data,feature)
        
        
        self.H = np.sign(np.dot(alpha_t,haar_feature_test_data))
        error_t = np.sum(self.H != combined_face_data_weight)
        
        TP = np.sum(self.H[:num_data] == combined_face_data_weight[:num_data] )
        FP = np.sum(self.H[:num_data] != combined_face_data_weight[:num_data])
        TN = np.sum(self.H[num_data:] == combined_face_data_weight[num_data:] )
        FN = np.sum(self.H[num_data:] != combined_face_data_weight[num_data:] )
        
        print("True Positives: {}/100,".format(TP))
        print("False Positives: {}/100,".format(FP))
        print("True Negatives: {}/100,".format(TN))
        print("False Negatives: {}/100,".format(FN))
        len_d = 200//2
        actual = np.append([1]*len_d,[0]*len_d)
        
        # PLotting Actual vs Prediction
        false_p, true_p, _ = roc_curve(actual,np.asarray((self.H)))
        print("Accuracy is :",auc(false_p,true_p))
        
        plt.figure()
        plt.plot(false_p,true_p)
        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.title('ROC Curve with accuracy {}'.format(auc(false_p,true_p)))
        
        plt.show()
        
                    
                    
    def visualize_adaboost(self,ts_face_data,haar_idx,feature):
        
        coord_f, _ = haar_like_feature_coord(ts_face_data[1].shape[0], ts_face_data[1].shape[1],feature)
        
        HaarFeatureExtraction.visualize_haar_feature(ts_face_data[1],coord_f)
        
        for i in range(10):
            self.plot_haar_images(ts_face_data,coord_f[haar_idx[i]])
        
        
if __name__ == "__main__":
    
    feature_types = ['type-2-x', 'type-2-y','type-3-x','type-3-y','type-4']
    
    #Load Data
    face_train, nface_train, face_test, nface_test = load_pickle_data()

    # Compute HAAR features for face and non-face training data
    model_test = HaarFeatureExtraction()
    aa = model_test.get_haar_feature_list(face_train,feature_types)
    bb = model_test.get_haar_feature_list(nface_train,feature_types)
    
    combined_feature_data = np.vstack((aa,bb))
    
    # Compute threshold value for the HAAR features
    threshold_data =  model_test.get_threshold(face_train,nface_train,feature_types)
    
    # Compute Adaboost
    len_dataset = 500 
    num_epoch = 20
    ada_model = Adaboost(combined_feature_data,threshold_data,len_dataset)
    haar_index,alpha = ada_model.adaboost_compute(combined_feature_data,threshold_data,len_dataset, num_epoch)
    
    ts_f = list(face_test)
    ts_nf = list(nface_test)
    cm_ts_data = ts_f + ts_nf
    # Predict the model on Test data to find Accuracy score and ROC plots
    ada_model.predict(cm_ts_data,haar_index,alpha,feature_types)
    
    model_test.visualize_haar_feature(face_test,feature_types)