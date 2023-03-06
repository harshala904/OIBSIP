 #### Task 5 : Sales Prediction Using Python
 #### Name : Dhukate Harshala Gajendra
 #### Batch - February Phase 2 OIBSIP
 ### Step 1 : Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
df = pd.read_csv("C:/Users/a/Downloads/Advertising.csv")
df
df.head()
df.tail()
df.shape
df.info()
#check null values
df.isnull().sum()
# #### Describing Data
df.describe()
df.corr()
### Step 3: Exploratory data analysis
sns.heatmap(df.corr(), square=True)
sns.lmplot(x='TV', y='Sales', data=df)
df.hist()
plt.show()
# ###   Step 4 : model fitting
X = df[['TV']]
y = df.Sales
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
y_pred = model.predict(X_test)
act_predict= pd.DataFrame({
    'Actual': y_test.values.flatten(), 
    'Predict': y_pred.flatten()})
act_predict.head(10)
act_predict.sample(10).plot(kind='hist')





