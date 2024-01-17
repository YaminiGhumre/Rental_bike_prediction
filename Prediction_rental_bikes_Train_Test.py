#!/usr/bin/env python
# coding: utf-8

# In[1]:


###In Exploratory Data Analysis, we first look at the following:

###Summary of data
#Unique values of each column
#Whether there are missing values
#First we import all relevant libraries:


# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px ## to generate the bar graph
from datetime import datetime, date
from sklearn import preprocessing, metrics, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor


# In[3]:


train = pd.read_csv('E:/Python docs/train.csv')
print(train.head(5))


# In[4]:


train.info()


# In[5]:


train.describe().T


# In[6]:


##Based on info and description of the training dataframe as above, there is no missing data across the columns in the training dataframe.


# In[7]:


missing_values = pd.DataFrame(train.isna().sum(), columns = ['Missing Values'])
missing_values


# In[ ]:





# In[8]:


# Next, we split datetime into separate year, month, day, hour and dayofweek columns


# In[9]:


train['yyyymmdd'] = train['datetime'].apply(lambda x : x.split()[0])
train['year'] = train['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').year)
train['month'] = train['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').month)
train['date'] = train['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').day)
train['hour'] = train['datetime'].apply(lambda x : x.split()[1].split(":")[0])
train = train.drop(['datetime', 'yyyymmdd'], axis = 1)


# In[10]:


train.columns


# ### Day Column Into Week

# In[11]:


week = []
for i in train['date']:
    if i < 8:
        week.append(1)
    elif i >= 8 and i < 16:
        week.append(2)
    elif i >=16 and i < 22:
        week.append(3)
    else:
        week.append(4)
train['week'] = week


# ### Converting Hour Column to Int Type

# In[12]:


train['hour'] = train['hour'].astype('object').astype('int64')


# In[13]:


train.columns


# In[14]:


##With the data cleaned, we look at the correlation coefficient between each feature pairing to look at how correlated each pair of feature is:


# In[15]:


fig, ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(train.corr(), vmax = 1, vmin = -1, square = False, annot = True, mask = np.triu(train.corr()))


# In[16]:


#Count is very highly positively related to registered and casual.
#Season is very highly positively related to month.
#Temperature is very highly positively related to actual temperature
#In addition, we know that casual + registered = count. Hence we can drop columns ‘registered’, ‘casual’, and ‘season’.


# # Season

# In[17]:


train.groupby('season')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Season', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# # Holiday

# In[18]:


train.groupby('holiday')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Holiday', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[19]:


###     98% bookings are done on non-holidays.


# # Workingday
# 

# In[20]:


train.groupby('workingday')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Working Day', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[21]:


##69% bookings are done on working days.


# # weather

# In[22]:


train.groupby('weather')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('weather', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[23]:


### most of the booking done in weather 1 


# # Actual Temprature

# In[24]:


sns.regplot(x =train['atemp'], y =train['count'], line_kws = {'color': 'red'})
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Actual Temperature', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[25]:


## booking are increased with increased in temprature


# In[26]:


sns.boxplot(y = train['atemp']) ### to see if there is any outliners
plt.title('Train Actual Temperature')


# # Humidity

# In[27]:


sns.regplot(x =train['humidity'], y = train['count'], line_kws = {'color': 'red'})
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Humidity', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[28]:


###    Bookings are decreasing with the increase in humidity.


# In[29]:


sns.boxplot(y =train['humidity'])
plt.title('Train Humidity')


# # Wind speed

# In[30]:


sns.regplot(x =train['windspeed'], y = train['count'], line_kws = {'color': 'red'})
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Wind Speed', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[31]:


####   Bookings are slightly increasing with the increase in wind speed.


# In[32]:


sns.boxplot(y =train['windspeed'])
plt.title('Train Wind Speed')


# ### Removing outliners from Windspeed

# In[33]:


wind_speed_train = []
for i in train['windspeed']:
    if i < (train['windspeed'].mean() - (2 * train['windspeed'].std())):
        wind_speed_train.append(train['windspeed'].mean() - (2 * train['windspeed'].std()))
    elif i > (train['windspeed'].mean() + (2 * train['windspeed'].std())):
        wind_speed_train.append(train['windspeed'].mean() + (2 * train['windspeed'].std()))
    else:
        wind_speed_train.append(i)
train['windspeed'] = wind_speed_train


# In[34]:


sns.boxplot(y =train['windspeed'])
plt.title('Train Wind Speed')


# # Year

# In[35]:


train.groupby('year')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[36]:


###This data is not that valuable as we don't have the complete 2012 data with us. It is just upto June 2012.
##However we can compare the first six months of these two years.


# In[37]:


new_df = train[train['month'] < 7]
new_df.groupby('year')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[38]:


### more booking are done in year 2012 as compared to 2011 in the first 6 months


# # Month

# In[39]:


train.groupby('month')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[40]:


### majorly booking done from March - June


# # WEEK

# In[41]:


train.groupby('week')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('week', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# ### Hour

# In[42]:


train.groupby('hour')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Hour', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# ### Count

# In[43]:


sns.boxplot(y = train['count'])
plt.title('Train Count')


# ### Removing Outliners from Count

# In[44]:


count_train = []
for i in train['count']:
    if i < (train['windspeed'].mean() - (2 * train['windspeed'].std())):
        count_train.append(train['windspeed'].mean() - (2 * train['windspeed'].std()))
    elif i > (train['windspeed'].mean() + (2 * train['windspeed'].std())):
        count_train.append(train['windspeed'].mean() + (2 * train['windspeed'].std()))
    else:
        count_train.append(i)
train['count'] = count_train


# ### Dropping Unnecessary Columns

# In[45]:


##train = train.drop(['temp', 'casual', 'registered','month'], axis = 1)


# In[46]:


train_scaled = pd.DataFrame(StandardScaler().fit_transform(train.drop('count', axis = 1)), columns =train.drop('count', axis = 1).columns)


# In[47]:


train_scaled.head()


# In[48]:


X = train_scaled[['hour', 'atemp', 'workingday', 'humidity', 'season']]
y =train['count']


# In[49]:


X_train,X_test,y_train,y_test = train_test_split( X, y, test_size = 0.2, random_state= 1234)


# In[50]:


(X_train.shape),(X_test.shape),(y_train.shape),(y_test.shape)


# In[51]:


lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)


# In[52]:


print('Train R2 : ', r2_score(y_train, y_train_pred))
print('Validation R2 : ', r2_score(y_test, y_test_pred))


# In[53]:


sns.distplot(y_test_pred - y_test)


# ### KNN Regression Model-
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=123)

# In[54]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


# In[55]:


model = KNeighborsRegressor(n_neighbors=5) 
print("The model is loaded")


# In[56]:


### training the model
model_fitting = model.fit(X_train, y_train)
print("Model training is completed")


# In[57]:


#### Getting the training score
model_fitting.score(X_train, y_train)


# In[58]:


#### Prediction of testing data
pred = model_fitting.predict(X_test)
results=r2_score(y_test,pred)
print(results)


# In[59]:


get_ipython().system('pip install import-ipynb')


# In[60]:


def function_calling():  ##### Parent Function
    X, y = create_data(1000,2)
    X_train,X_test,y_train,y_test = train_test(X,y)
    return X_train,X_test,y_train,y_test


# ### Testing Data

# In[61]:


df_test = pd.read_csv('E:/Python docs/test.csv')
print(df_test.head(5))


# In[62]:


df_test['yyyymmdd'] = df_test['datetime'].apply(lambda x : x.split()[0])
df_test['year'] = df_test['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').year)
df_test['month'] = df_test['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').month)
df_test['date'] = df_test['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').day)
df_test['hour'] = df_test['datetime'].apply(lambda x : x.split()[1].split(":")[0])
df_test = df_test.drop(['datetime', 'yyyymmdd'], axis = 1)
week = []
for i in df_test['date']:
    if i < 8:
        week.append(1)
    elif i >= 8 and i < 16:
        week.append(2)
    elif i >=16 and i < 22:
        week.append(3)
    else:
        week.append(4)
df_test['week'] = week
df_test['hour'] = df_test['hour'].astype('object').astype('int64')
df_test['windspeed'] = df_test['windspeed'] ** (1/2)
wind_speed_test = []
for i in df_test['windspeed']:
    if i < (df_test['windspeed'].mean() - (2 * df_test['windspeed'].std())):
        wind_speed_test.append(df_test['windspeed'].mean() - (2 * df_test['windspeed'].std()))
    elif i > (df_test['windspeed'].mean() + (2 * df_test['windspeed'].std())):
        wind_speed_test.append(df_test['windspeed'].mean() + (2 * df_test['windspeed'].std()))
    else:
        wind_speed_test.append(i)
df_test['windspeed'] = wind_speed_test
df_test = df_test.drop(['temp', 'casual', 'registered', 'date'], axis = 1)
df_test_scaled = pd.DataFrame(StandardScaler().fit_transform(df_test), columns = df_test.columns)
X_test = df_test_scaled[['hour', 'atemp', 'workingday', 'humidity', 'season']]


# In[65]:


X_test.head(5)


# ### Final Prediction

# In[74]:


from sklearn.neighbors import KNeighborsRegressor
KNN_model = KNeighborsRegressor(n_neighbors=5).fit(X_train,y_train)
Final_pred = KNN_model.predict(X_test)
print(Final_pred)


# ### Exporting Predictions to CSV

# In[75]:


Final_pred = pd.DataFrame((Final_pred) ** 3, columns = ['Predicted Counts'])
Final_pred = Final_pred.round(decimals = 0)
Final_pred['Predicted Counts'] = Final_pred['Predicted Counts'].astype('float').astype(int)
Final_pred.to_csv("Final_pred")


# In[76]:


Final_pred.to_csv("E:/python docs" + "Final_pred.csv")


# In[ ]:





# In[ ]:





# In[ ]:




