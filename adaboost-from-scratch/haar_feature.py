# -*- coding: utf-8 -*-
"""
@author: asmathew@ncsu.edu
"""

# Change path according to directory
import sys
sys.path.append('C:/Users/Aaron/Desktop/ECE 763 project/asmathew_project02')
from dataset.load_data import load_pickle_data

import numpy as np
import math
from tqdm import tqdm
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, haar_like_feature_coord,draw_haar_like_feature
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import cv2 

from tqdm import tqdm

class HaarFeatureExtraction():
    
    def __init__(self):
        
        self.img = None
        self.feature = None
        
    def get_haar_feature(self,img,feature):

        image_integral = integral_image(img)
        
        return haar_like_feature(image_integral, 0, 0,image_integral.shape[0], image_integral.shape[1],feature)
    
    def get_haar_feature_coord(self,img,feature):
        
        feature_coord, feature_type = haar_like_feature_coord(img.shape[0], height=img.shape[1],
                            feature_type=feature)
        
        return feature_coord
    
    def get_haar_feature_list(self,dataset,feature):
        
        haar_feature_list = [self.get_haar_feature(data,feature) for data in tqdm(dataset,desc="{Computing HAAR features}")]
        
        return np.asarray(haar_feature_list)
    
    def get_threshold(self,face_data,nface_data,feature):
               
             
        thresh_idx = []
        N = 1000
        
        haar_face_data = self.get_haar_feature_list(face_data,feature) 
        haar_nface_data = self.get_haar_feature_list(nface_data,feature)

        threshold = (haar_face_data.mean(axis=0) + haar_nface_data.mean(axis=0))/2 
        return np.asarray(threshold)
    
    
    def visualize_haar_feature(self,images,feature):
        fig, axs = plt.subplots(3, 2)
        images = list(images)
        #coord, _ = haar_like_feature_coord(images[2].shape[0], images[2].shape[1], [feat_t])
        for ax, feat_t in zip(np.ravel(axs),feature):
            coord, _ = haar_like_feature_coord(images[2].shape[0], images[2].shape[1], [feat_t])
            haar_feature = draw_haar_like_feature(images[2], 0, 0,
                                                  images[2].shape[0],
                                                  images[2].shape[1],
                                                  coord,
                                                  max_n_features=1,
                                                  random_state=0)
        
            ax.imshow(haar_feature)
            ax.set_title(feat_t)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle('HAAR Features')
        plt.axis('off')
        plt.imshow(images[2])
        plt.show()
    

        