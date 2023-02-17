 #### Task 1  :  Train Machine Learning Model For a Iris Dataset
# #### Name : Dhukate Harshala Gajendra
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/a/Downloads/Iris (1).csv")
df.head()
df.shape
df.describe()
df.info()
df.isnull().sum()
df1=df[["SepalLengthCm" , "SepalWidthCm" , "PetalLengthCm" , "PetalWidthCm" , "Species"]]
print(df1.head())
# ###  Step 3 :  Explortary Data Analysis
fig, axes = plt.subplots(2, 2, figsize=(15,15))
 axes[0,0].set_title("SepalLengthCm\n",fontsize=15,color="navy")
axes[0,0].hist(df['SepalLengthCm'], bins=7)
 axes[0,1].set_title("SepalWidthCm\n",fontsize=15,color="navy")
axes[0,1].hist(df['SepalWidthCm'], bins=5);
 axes[1,0].set_title("PetalLengthCm\n",color="cornflowerblue",fontsize=15)
axes[1,0].hist(df['PetalLengthCm'], bins=6);
 axes[1,1].set_title("PetalWidthCm\n",fontsize=15,color="lavender")
axes[1,1].hist(df['PetalWidthCm'], bins=6);
df['Species'].hist()
#scraterplot
colors =['red','orange','blue']
Species = ['Iris-setosa','Iris-versicolor','Iris-virginica']
for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i],label=Species[i])
    plt.xlabel("SepalLengthCm")
    plt.ylabel("SepalWidthCm")
for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i],label=Species[i])
    plt.xlabel("PetalLengthCm")
    plt.ylabel("PetalWidthCm")
for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i],label=Species[i])
    plt.xlabel("SepalLengthCm'")
    plt.ylabel("PetalLengthCm")
for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i],label=Species[i])
    plt.xlabel("SepalWidthCm")
    plt.ylabel("PetalWidthCm")
df.corr()
# #Show in matrics form
corr =df.corr()
fig,ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head()
# ### Step 4 : Building the ML model
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
# #decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print("Accuracy:",model.score(x_test,y_test)*100)
# #KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)*100)
# #RandomForest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)*100)





