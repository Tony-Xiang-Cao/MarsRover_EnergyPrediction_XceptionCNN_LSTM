# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:05:37 2020

@author: Xiang Cao

The pre-processing functions for 
loading images and power from dataset
and loading pre-trained model
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras



#Use opencv to pre-process images
def process_image(directory_name):
    img_list= []
    for filename in os.listdir(r'./'+directory_name):
        img= cv2.imread(directory_name+'/'+filename)
        img_crop = img[280:,52:752] #crop top half, and left&right side of the image
        img_resize= cv2.resize(img_crop, (300,200)) #horizontally shrink 50%
        img_norm = np.zeros((300,200))
        img_norm = cv2.normalize(img_resize,img_norm, 0,255,cv2.NORM_MINMAX)
        img_list.append(img_norm)
    return img_list


#load images from the dataset
def load_image (directory_name):
    train_image = np.asarray(process_image(directory_name))
    print('Shape of training images of ', directory_name,' are : ' ,train_image.shape)
    return train_image


#function for loading power data from energy.txt
def load_power(input_dir,start,end):
    energy_str= 'energy.txt'
    f=open(os.path.join(input_dir,energy_str) ,"r")
    lines=f.readlines()
    p_l=[] #left wheel power
    p_r=[] #righ wheel power
    for x in lines[1:]:
        str_l= x.split(',')[5]
        flt_l=float(str_l) #conver string to float
        p_l.append(flt_l)
        str_r= x.split(',')[6]
        flt_r= float(str_r) #conver string to float
        p_r.append(flt_r)

    total_power=np.add(p_l,p_r)
    
    DELETE_STEP=9 
    # The rgb images are 1/9 less than the energy data
    # Remove 1 power sample among every 9 samples
    power_delete= np.delete(total_power,np.arange(0, total_power.size,DELETE_STEP)) 
    #remove several starting and ending power data, 
    #to match the sample amount of color images
    #Also starting and ending power values are zero when rover are stand still.
    power_data = np.expand_dims(power_delete, axis=1)[start:end]
    print('Shape of power data of ', input_dir,' are : ' ,power_data.shape)
    
    return power_data


'''
    This model is trained in the IPython notebook Xception_LSTM_TimeDistributed-Model.ipynb
    The best performing model with lowest validation MAE 
    is saved during training using callback functions
'''
def load_model():
    print("For detail training model, please read Xception_LSTM_TimeDistributed-Model.ipynb")
    saved_model = keras.models.load_model('./src/bestXceptionModel')
    print(saved_model.summary())
    return saved_model


