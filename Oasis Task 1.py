#!/usr/bin/env python
# coding: utf-8

# ### Oasis Infobytes : Data Science Internship 

# #### Task 1  :  Train Machine Learning Model For a Iris Dataset

# #### Name : Dhukate Harshala Gajendra

# ### Step 1 : Importing Libraries

# In[31]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ### step 2 : Load The Dataset 

# In[32]:


df = pd.read_csv("C:/Users/a/Downloads/Iris (1).csv")
df.head()


# In[33]:


df.shape


# In[34]:


#To display stats about data
df.describe()


# In[35]:


df.info()


# In[36]:


#check null values
df.isnull().sum()


# In[37]:


df1=df[["SepalLengthCm" , "SepalWidthCm" , "PetalLengthCm" , "PetalWidthCm" , "Species"]]
print(df1.head())


# ###  Step 3 :  Explortary Data Analysis

# In[38]:


fig, axes = plt.subplots(2, 2, figsize=(15,15))
 
axes[0,0].set_title("SepalLengthCm\n",fontsize=15,color="navy")
axes[0,0].hist(df['SepalLengthCm'], bins=7)
 
axes[0,1].set_title("SepalWidthCm\n",fontsize=15,color="navy")
axes[0,1].hist(df['SepalWidthCm'], bins=5);
 
axes[1,0].set_title("PetalLengthCm\n",color="cornflowerblue",fontsize=15)
axes[1,0].hist(df['PetalLengthCm'], bins=6);
 
axes[1,1].set_title("PetalWidthCm\n",fontsize=15,color="lavender")
axes[1,1].hist(df['PetalWidthCm'], bins=6);


# In[39]:


df['Species'].hist()


# In[40]:


#scraterplot
colors =['red','orange','blue']
Species = ['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[41]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i],label=Species[i])
    plt.xlabel("SepalLengthCm")
    plt.ylabel("SepalWidthCm")


# In[42]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i],label=Species[i])
    plt.xlabel("PetalLengthCm")
    plt.ylabel("PetalWidthCm")


# In[43]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i],label=Species[i])
    plt.xlabel("SepalLengthCm'")
    plt.ylabel("PetalLengthCm")


# In[44]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i],label=Species[i])
    plt.xlabel("SepalWidthCm")
    plt.ylabel("PetalWidthCm")


# In[45]:


df.corr()


# #Show in matrics form

# In[46]:


corr =df.corr()
fig,ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax)


# In[47]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[48]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# ### Step 4 : Building the ML model

# In[49]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# #decision tree

# In[50]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[51]:


model.fit(x_train, y_train)


# In[52]:


#print metrics
print("Accuracy:",model.score(x_test,y_test)*100)


# #KNN

# In[57]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[58]:


model.fit(x_train,y_train)


# In[59]:


print("Accuracy:",model.score(x_test,y_test)*100)


# #RandomForest

# In[60]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


model=RandomForestClassifier()
model.fit(x_train,y_train)


# In[62]:


print("Accuracy:",model.score(x_test,y_test)*100)


# #### Conclusion

# The decision tree model of accuracy is  97.77777777777777

# The KNeighborsClassifier model of accuracy is  100.

# The RandomForest model of accuracy is  100.

# ### Thank You

# In[ ]:




