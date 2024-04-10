#!/usr/bin/env python
# coding: utf-8

# In[19]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[20]:


(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
x_train.shape


# In[21]:


x_test.shape


# In[22]:


y_train.shape


# In[23]:


y_train=y_train.reshape(-1,)
y_train[:5]


# In[24]:


y_test = y_test.reshape(-1,)


# In[25]:


classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# In[28]:


def plot_sample(x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])


# In[29]:


plot_sample(x_train, y_train, 0)


# In[31]:


plot_sample(x_train, y_train, 1)


# In[33]:


x_train = x_train / 255.0 #Normalize the images to a number from 0 to 1. Image has 3 channels (R,G,B) and each value in the channel can range from 0 to 255. Hence to normalize in 0-->1 range, we need to divide it by 255
x_test = x_test / 255.0


# In[35]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(x_train, y_train, epochs=5)


# In[37]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(x_train, y_train, epochs=10)


# In[ ]:




