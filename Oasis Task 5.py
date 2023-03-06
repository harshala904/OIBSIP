#!/usr/bin/env python
# coding: utf-8

# ### Oasis Infobytes : Data Science Internship 

# #### Task 5 : Sales Prediction Using Python

# #### Name : Dhukate Harshala Gajendra

# #### Batch - February Phase 2 OIBSIP

# ### Step 1 : Importing Libraries

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression


# ### step 2 : LoadingThe Dataset 

# In[66]:


df = pd.read_csv("C:/Users/a/Downloads/Advertising.csv")
df


# In[67]:


df.head()


# In[68]:


df.tail()


# In[69]:


df.shape


# In[70]:


df.info()


# In[71]:


#check null values
df.isnull().sum()


# #### Describing Data

# In[72]:


df.describe()


# In[73]:


df.corr()


# ### Step 3: Exploratory data analysis

# In[74]:


sns.heatmap(df.corr(), square=True)


# In[75]:


sns.lmplot(x='TV', y='Sales', data=df)


# In[76]:


df.hist()
plt.show()


# ###   Step 4 : model fitting

# In[77]:


X = df[['TV']]
y = df.Sales


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

model = LinearRegression()
model.fit(X_train, y_train)


# In[79]:


print(model.coef_)
print(model.intercept_)


# In[80]:


y_pred = model.predict(X_test)
act_predict= pd.DataFrame({
    'Actual': y_test.values.flatten(), 
    'Predict': y_pred.flatten()})


# In[81]:


act_predict.head(10)


# In[83]:


act_predict.sample(10).plot(kind='hist')


# ###  Thank You

# In[ ]:




