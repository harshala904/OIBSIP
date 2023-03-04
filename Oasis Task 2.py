# #### Task 2  : Unemployment  Analysis With Python
# #### Name : Dhukate Harshala Gajendra
# #### Batch - February Phase 2 OIBSIP
# ### Step 1 : Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# ### step 2 : LoadingThe Dataset 
df = pd.read_csv("C:/Users/a/Downloads/unemployment rate.csv")
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
# ### Step 3: Exploratory data analysis
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(14,12))
sns.heatmap(df.corr())
plt.show()
#Boxplot
df[["States","Date","Frequency",
               "Estimated Unemployment Rate","Estimated Employed",
               "Estimated Labour Participation Rate","Region",
               "longitude","latitude"]]
df.boxplot()
import seaborn as sns
sns.pairplot(df)
df.hist()
plt.show()






