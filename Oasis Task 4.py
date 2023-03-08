#!/usr/bin/env python
# coding: utf-8

# ### Oasis Infobytes : Data Science Internship 

# #### Task 4  : Email Spam Detection With Machine Learning

# #### Name : Dhukate Harshala Gajendra

# #### Batch - February Phase 2 OIBSIP

# ### Step 1 : Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ### step 2 :  Loading Dataset

# In[2]:


data = pd.read_csv("C:/Users/a/Downloads/spam.csv",encoding="latin1")
data


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


#check null values
data.isnull().sum()


# In[8]:


data.describe()


#  ### Step 3: data preprocessing

# In[9]:


del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']


# In[10]:


data.rename(columns = {'v1':'Category', 'v2':'Message'}, inplace = True)


# In[11]:


spam = data.groupby('Category')['Message'].count()
spam


# In[14]:


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
data['spam'] = label_encoder.fit_transform(data['Category'])
data.head()


# In[16]:


x = data['Message']
y = data['spam']


#  ### Exploratary Data Analysis

# In[36]:


sb.histplot(y)


# In[41]:


sb.barplot(y)


# ### Data splitting

# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()


# In[19]:


# Email converted into number matrix

x_train_count = v.fit_transform(x_train.values)
x_train_count.toarray()[:3]


# ### Model selection and implimentation  

# In[20]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()


# In[21]:


model.fit(x_train_count, y_train)


# In[22]:


x_test_count = v.transform(x_test)
model.predict(x_test_count)


# In[24]:


pred = model.predict(x_test_count)


# In[25]:


pred


# In[45]:


from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, pred)
cm


# In[53]:


model.score(x_test_count, y_test)


#  ### Thank  You
