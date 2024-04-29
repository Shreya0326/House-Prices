#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Reading the dataset (train.csv)

# In[2]:


data = pd.read_csv('housing price.csv')


# In[3]:


data.head()


# ## Exploratory Data Analysis

# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.select_dtypes(include='object').columns


# In[8]:


len(data.select_dtypes(include='object').columns)


# In[9]:


data.select_dtypes(include=['int64','float64']).columns


# In[10]:


len(data.select_dtypes(include=['int64','float64']).columns)


# ## Dealing with null values

# In[11]:


# Checking the null value or not
data.isnull().values.any()


# In[12]:


data.isnull().values.sum()


# In[13]:


data.isnull().sum()


# In[14]:


# Finding the columns having null values
error_col = data.columns[data.isnull().any()]
error_col


# In[15]:


# Representing the columns containing error
error_data = data[error_col]
error_count = error_data.isnull().sum()
print(error_count)


# In[16]:


# showing this by graph
plt.figure(figsize=(10,6))
error_count.plot(kind='bar',color="pink")
plt.title("Error counts in coloumns")
plt.show()


# In[17]:


# checking the shape again 
data.shape


# In[18]:


# checking the null value in percentage value
error_count_per = error_data.isnull().mean()*100
error_count_per


# In[19]:


# select the coloumns which has more than 60%
data_drop = error_count_per[error_count_per > 60].keys()


# In[20]:


data_drop


# In[21]:


data = data.drop(labels=data_drop,axis=1)


# In[22]:


data.shape


# In[23]:


# checking the null value again
data.columns[data.isnull().any()]


# In[24]:


len(data.columns[data.isnull().any()])


# Add column mean in numerical column

# In[25]:


# fist find the numeriacl coloumns and handle the missing value


# In[26]:


numerical_data = set(data.select_dtypes(include = ['int64','float64']).columns).intersection(data.columns[data.isnull().any()])


# In[27]:


numerical_data


# In[28]:


data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean())
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mean())


# In[29]:


len(data.columns[data.isnull().any()])


# In[ ]:





# In[30]:


# lets see the remaining coloumn (i.e in object) for dropping


# In[31]:


data.columns[data.isnull().any()]


# In[32]:


data['MasVnrType'] =  data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])
data['BsmtQual'] = data['BsmtQual'].fillna(data['BsmtQual'].mode()[0])
data['BsmtCond'] = data['BsmtCond'].fillna(data['BsmtCond'].mode()[0])
data['BsmtExposure'] = data['BsmtExposure'].fillna(data['BsmtExposure'].mode()[0])
data['BsmtFinType1'] = data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0])
data['BsmtFinType2'] = data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0])
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data['FireplaceQu'] = data['FireplaceQu'].fillna(data['FireplaceQu'].mode()[0])
data['GarageType'] = data['GarageType'].fillna(data['GarageType'].mode()[0])
data['GarageFinish'] = data['GarageFinish'].fillna(data['GarageFinish'].mode()[0])
data['GarageQual'] = data['GarageQual'].fillna(data['GarageQual'].mode()[0])
data['GarageCond'] = data['GarageCond'].fillna(data['GarageCond'].mode()[0])


# In[33]:


len(data.columns[data.isnull().any()])


# In[34]:


# check the null values again, exist or not
data.isnull().values.any()


# In[35]:


# count of null values
data.isnull().values.sum()


# In[36]:


# The given data is
data.head()


# In[37]:


# describe the target


# In[38]:


data['SalePrice'].describe()


# In[39]:


import warnings
warnings.filterwarnings('ignore')


# In[40]:


# plot the displot of target value
plt.figure(figsize=(16,9))
bar = sns.distplot(data['SalePrice'])
bar.legend(['Skewness: {:.2f}'.format(data['SalePrice'].skew())])
plt.show()


# In[41]:


# Correlation matrix or Heatmap


# In[42]:


data_1 = data.drop(columns ='SalePrice')


# In[43]:


data_1.corrwith(data['SalePrice']).plot.bar(
    figsize=(16,9),title = 'Correlation with saleprice',
    rot = 45,grid = True   
)


# In[44]:


# Correlation matrix
plt.figure(figsize=(25,30))
ax = sns.heatmap(data.corr(),cmap='coolwarm',annot = True,linewidths=2)


# In[45]:


# Correlation heatmap of highly correlated features with 'SalePrice'


# In[46]:


high_corr = data.corr()


# In[47]:


high_corr_features = high_corr.index[abs(high_corr['SalePrice']) >= 0.5]


# In[48]:


high_corr_features


# In[49]:


len(high_corr_features)


# In[50]:


# Correlation heatmap of highly correlated features with 'SalePrice'


# In[51]:


plt.figure(figsize = (16,9))
ax = sns.heatmap(data[high_corr_features].corr(),cmap='coolwarm',annot = True,linewidths=2)


# ## Dealing with catregorical data

# In[52]:


data.shape


# In[53]:


# Categorical columns


# In[54]:


data.select_dtypes(include = 'object').columns


# In[55]:


len(data.select_dtypes(include = 'object').columns)


# In[56]:


# Do one hot encoding
data = pd.get_dummies(data=data,drop_first=True)


# In[57]:


data.shape


# In[58]:


# Checking the categorical columns
data.select_dtypes(include = 'object').columns


# In[59]:


# checking the length
len(data.select_dtypes(include='object').columns)


# ## Splitting the data

# In[60]:


# Independent data
x = data.drop(columns='SalePrice')


# In[61]:


# Dependent data
y = data['SalePrice']


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state=0)


# In[64]:


x_train.shape


# In[65]:


y_train.shape


# In[66]:


x_test.shape


# In[67]:


y_test.shape


# ## Feature Scaling

# In[68]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[69]:


x_train


# In[70]:


x_test


# ##  Building the model

# ### Multiple Linear Regression

# In[71]:


from sklearn.linear_model import LinearRegression
regressor_mlr = LinearRegression()
regressor_mlr.fit(x_train,y_train)


# In[72]:


y_pred = regressor_mlr.predict(x_test)


# In[73]:


from sklearn.metrics import r2_score


# In[74]:


r2_score(y_test,y_pred)


# ### Random Forest Regression

# In[75]:


from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor()
regressor_rf.fit(x_train,y_train)


# In[76]:


y_pred = regressor_rf.predict(x_test)


# In[77]:


from sklearn.metrics import r2_score


# In[78]:


r2_score(y_test,y_pred)


# ### XGBoost regression

# In[79]:


from xgboost import XGBRFRegressor
regressor_xgb = XGBRFRegressor()
regressor_xgb.fit(x_train,y_train)


# In[80]:


y_pred = regressor_xgb.predict(x_test)


# In[81]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# ##  Hyper parameter tuning

# In[82]:


from sklearn.model_selection import RandomizedSearchCV


# In[83]:


parameters = {
    'n_estimators':[200,400,600,800,1000,1200,1400,1600,],
    'max_depth':[10,20,30,40,50,60,70,80,90,100,None],
    'min_samples_split': [2,5,10],
    'min_samples_leaf':[1,2,4],
    'max_features':['auto','sqrt'],
    'bootstrap':[True,False]  
        
}


# In[85]:


parameters


# In[86]:


random_cv = RandomizedSearchCV(estimator=regressor_rf,param_distributions=parameters,n_iter=20,
                               cv=5,verbose=2,n_jobs=-1,random_state=0)


# In[87]:


random_cv.fit(x_train, y_train)


# In[88]:


random_cv.best_estimator_


# In[90]:


random_cv.best_params_


# ##  Final model (Random forest regressor)

# In[91]:


from sklearn.ensemble import RandomForestRegressor


# In[97]:


regressor = XGBRFRegressor(base_score=None, booster=None, callbacks=None,
               colsample_bylevel=None, colsample_bytree=None, device=None,
               early_stopping_rounds=None, enable_categorical=False,
               eval_metric=None, feature_types=None, gamma=None,
               grow_policy=None, importance_type=None,
               interaction_constraints=None, max_bin=None,
               max_cat_threshold=None, max_cat_to_onehot=None,
               max_delta_step=None, max_depth=None, max_leaves=None,
               min_child_weight=None, missing=float('nan'), monotone_constraints=None,
               multi_strategy=None, n_estimators=None, n_jobs=None,
               num_parallel_tree=None, objective='reg:squarederror',
               random_state=None, reg_alpha=None)
regressor.fit(x_train, y_train)


# In[98]:


y_pred = regressor.predict(x_test)


# In[100]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:




