#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import math
import os
from PIL import Image
import os
from tensorflow.keras import Sequential,Model,models
from tensorflow.keras.layers import Dense,Dropout,Flatten,BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image



# In[2]:


import streamlit as st
def predict_age(file):
  model1=tf.keras.models.load_model("age_vgg16.h5")
  img=image.load_img(file,target_size=(220,220))
  img=image.img_to_array(img)
  img=img/255.0
  img = np.expand_dims(img, axis=0)
  img_class=model1.predict(img)
  val=img_class[0]
  val=math.floor(val)
  return val

def predict_gender(file):
  model2=tf.keras.models.load_model("model_gender.h5")
  img=image.load_img(file,target_size=(220,220))
  img=image.img_to_array(img)
  img=img/255.0
  img = np.expand_dims(img, axis=0)
  img_class=model2.predict(img)
  val=img_class[0]
  return val
  


# In[3]:


st.title("PANORAMIC RADIOGRAPH ANALYSER")
st.write("TO PREDICT AGE AND GENDER")
file = st.file_uploader("Upload dental image", type=["jpg", "png"])
if file:
    st.image(file)
    #classify = st.button("Predict age")
    if st.button("predict age and gender"):
        st.write("Predicting age and gender...")
        age_label = predict_age(file)
        #(rem=age_label%10
        #if rem<= 5:
            #range1=age_label-rem
            #ange2=range1+5
        #else:
            #range3=age_label-rem
            #range1=range3+6
            #range2=range1+4)
            
        age_range = ((age_label - 1) // 10) * 10 + 1  # Calculate the lower bound of the age range
        age_range_upper = age_range + 9  # Calculate the upper bound of the age range

    gender_label = predict_gender(file)
    if gender_label >= 0.5:
    st.write("Age prediction: %d-%d \nGender prediction: Male" % (age_range, age_range_upper))
    else:
    st.write("Age prediction: %d-%d \nGender prediction: Female" % (age_range, age_range_upper))



# In[ ]:




