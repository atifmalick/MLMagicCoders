#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


df=pd.read_csv('F:\\5th Semester\\ML\\Python\\Multivariate_Regression\\Book1.csv')
df


# In[10]:


import math
median_bedroom=math.floor(df.bedroom.median())
median_bedroom


# In[12]:


df.bedroom=df.bedroom.fillna(median_bedroom)


# In[13]:


df


# In[15]:


reg = linear_model.LinearRegression()
reg.fit(df[['area','bedroom','age']],df.price)


# In[17]:


reg.predict([[3000,4,15]])


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Area,age,bedroom')
plt.ylabel('Price')
plt.scatter(df.area,df.price,df.bedroom,color='black', marker='*')
plt.plot(df.area,reg.predict(df[['area']]))


# In[ ]:




