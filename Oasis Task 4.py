# ### Oasis Infobytes : Data Science Internship 
# #### Task 4  : Email Spam Detection With Machine Learning
# #### Name : Dhukate Harshala Gajendra
# #### Batch - February Phase 2 OIBSIP
# ### Step 1 : Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# ### step 2 :  Loading Dataset
data = pd.read_csv("C:/Users/a/Downloads/spam.csv",encoding="latin1")
data
data.head()
data.tail()
data.shape
data.info()
#check null values
data.isnull().sum()
data.describe()
#  ### Step 3: data preprocessing
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data.rename(columns = {'v1':'Category', 'v2':'Message'}, inplace = True)
spam = data.groupby('Category')['Message'].count()
spam
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['spam'] = label_encoder.fit_transform(data['Category'])
data.head()
x = data['Message']
y = data['spam']
#  ### Exploratary Data Analysis
sb.histplot(y)
sb.barplot(y)
# ### Data splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
# Email converted into number matrix
x_train_count = v.fit_transform(x_train.values)
x_train_count.toarray()[:3]
# ### Model selection and implimentation  
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count, y_train)
x_test_count = v.transform(x_test)
model.predict(x_test_count)
pred = model.predict(x_test_count)
pred
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, pred)
cm
model.score(x_test_count, y_test)

