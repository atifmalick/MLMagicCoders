#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[26]:


df=pd.read_csv('F:\\5th Semester\\ML\\Python\\Multivariate_Regression\\Emp\\Book.csv')
df


# In[29]:


import math
median_experience=math.floor(df.experience.median())
median_testScore=math.floor(df.testScore.median())
df.experience=df.experience.fillna(median_experience)
df.testScore=df.testScore.fillna(median_testScore)
df


# In[30]:


reg=linear_model.LinearRegression()
reg.fit(df[['experience','testScore','interview']],df.salary)


# In[31]:


reg.predict([[2,9,10]])


# In[ ]:




