#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
df=pd.read_csv('F:\\5th Semester\\ML\\Python\\Decision_Tree\\Salaries\\salaries.csv')
df.head()


# In[8]:


input=df.drop('salary_more_than_100k',axis='columns')
target=df['salary_more_than_100k']


# In[11]:


from sklearn.preprocessing import LabelEncoder #for convert 


# In[13]:


#we have 3 columns so we have to create 3 objects
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()


# In[14]:


#Now create 3 new columns variable to store these object data
input['company_n']=le_company.fit_transform(input['company'])
input['hob_n']=le_company.fit_transform(input['job'])
input['degree_n']=le_company.fit_transform(input['degree'])


# In[15]:


input.head()


# In[21]:


#Drop original Columns
input_n=input.drop(['company','job','degree'],axis='columns')
input_n


# In[22]:


from sklearn import tree


# In[23]:


reg=tree.DecisionTreeClassifier()


# In[24]:


reg.fit(input_n,target)


# In[32]:


reg.predict([[1,1,1]])


# In[33]:


reg.score(input_n,target)


# In[ ]:




