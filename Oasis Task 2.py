#!/usr/bin/env python
# coding: utf-8

# ### Oasis Infobytes : Data Science Internship 

# #### Task 2  : Unemployment  Analysis With Python

# #### Name : Dhukate Harshala Gajendra

# #### Batch - February Phase 2 OIBSIP

# ### Step 1 : Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ### step 2 : LoadingThe Dataset 

# In[2]:


df = pd.read_csv("C:/Users/a/Downloads/unemployment rate.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


#check null values
df.isnull().sum()


# #### Describing Data

# In[8]:


df.describe()


# In[9]:


df.corr()


# ### Step 3: Exploratory data analysis

# In[10]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(14,12))
sns.heatmap(df.corr())
plt.show()


# In[14]:


#Boxplot

df[["States","Date","Frequency",
               "Estimated Unemployment Rate","Estimated Employed",
               "Estimated Labour Participation Rate","Region",
               "longitude","latitude"]]
df.boxplot()


# In[12]:


import seaborn as sns
sns.pairplot(df)


# In[27]:


df.hist()
plt.show()


# ### Thank You

# In[ ]:




