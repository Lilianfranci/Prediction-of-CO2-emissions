#!/usr/bin/env python
# coding: utf-8

# In[411]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

import warnings
warnings.filterwarnings('ignore')
#from pandas.plotting import scatter_matrix


# ### Data loading, cleaning and preparation

# In[129]:


#loading the dataset

fuelConsumption_df = pd.read_csv('MY2010-2014 Fuel Consumption Ratings 5-cycle (1).csv', encoding = '1250')
fuelConsumption_df.columns


# In[130]:


#dropping all unnnamed columns
fuelConsumption_df.dropna(axis='columns', inplace=True, how='all')


# In[131]:


#checking column names
fuelConsumption_df.columns


# In[132]:


#renaming columns 

fuelConsumption_df.rename(columns={'MODEL':"MODELYEAR",
                                   'MAKE':"MAKE", 
                                   'MODEL.1':'MODEL',
                                   'VEHICLE CLASS':'VEHCILECLASS',
                                   'ENGINE SIZE':'ENGINESIZE',
                                   'CYLINDERS':'CYLINDERS',
                                   'TRANSMISSION':'TRANSMISSION',
                                   'FUEL':'FUELTYPE', 
                                   'FUEL CONSUMPTION*':'FUELCONSUMPTION_CITY', 
                                   'Unnamed: 9':'FUELCONSUMPTION_HWY',
                                   'Unnamed: 10':'FUELCONSUMPTION_COMB', 
                                   'Unnamed: 11':'FUELCONSUMPTION_MPG',
                                   'CO2 EMISSIONS ':'CO2 EMISSIONS'}, inplace=True)


# In[133]:


#checking for null values 
fuelConsumption_df.isna().sum() 


# In[134]:


#dropping Null values as values are small compared to overrall dataset

fuelConsumption_df.dropna(inplace=True)
fuelConsumption_df.isnull().sum()


# In[135]:


#count and return sum of duplicates in df columns
fuelConsumption_df.duplicated().sum()


# In[136]:


fuelConsumption_df.info()


# In[137]:


#converting columns to appropriate datatypes

fuelConsumption_df = fuelConsumption_df.astype({'MODELYEAR':np.int, 'CYLINDERS':np.int, 'FUELCONSUMPTION_MPG':np.int, 'CO2 EMISSIONS': np.int, 'ENGINESIZE':np.float, 'FUELCONSUMPTION_CITY':np.float, 'FUELCONSUMPTION_HWY':np.float, 'FUELCONSUMPTION_COMB':np.float })
fuelConsumption_df.info()


# In[138]:


fuelConsumption_df.describe()


# ### Visualize

# In[139]:


#fuelConsumption_df.profile_report()


# In[140]:


# %matplotlib notebook

from matplotlib.backends import backend_agg
plt.figure(dpi=200)
fuelConsumption_df.hist(figsize = (20,15))
plt.show()
#plt.subplots_adjust(wspace = 0.5, hspace = 0.5)


# In[141]:


# showing the basic statistics of the dataset 

#fuelConsumption_df.describe().transpose()
fuelConsumption_df.describe().round(2)


# In[142]:


#checking for correlation between variables
fuelConsumption_df.corr()


# ## Split data into numeric and object

# In[156]:


fuelConsumption_df.info()


# In[192]:


# obtains all the numeric datatypes
numeric_df = fuelConsumption_df.select_dtypes(include=[np.number]) 
numeric_df


# In[145]:


# obtains all the datatypes that are objects
objects_df = fuelConsumption_df.select_dtypes(include = 'object')
objects_df


# ### Building model/ Scaling

# In[146]:


#dropping modelyear and CO2emissions column as its not required for computation, X is features and y is assigned to co2 target output

X =  numeric_df.drop(columns=['CO2 EMISSIONS','MODELYEAR' ])
y = numeric_df['CO2 EMISSIONS']


# In[147]:


X


# In[148]:


scaler = MinMaxScaler()


# In[149]:


X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
X_scaled


# In[426]:


plt.figure(dpi=200)

fig, ax = plt.subplots(figsize=(12,12))

numeric_df.drop(numeric_df[['MODELYEAR']], axis = 1)
sns.heatmap(numeric_df.corr(), annot= True, ax=ax)
plt.show()


# In[421]:


numeric_df.boxplot(figsize = (20,10), fontsize = 15, rot = 30)
plt.show()


# In[61]:


X.columns


# In[382]:


# for columns in X.columns:
#     sns.set(font_scale=2)
#     plt.figure(figsize=(20,15), dpi=200)
#     sns.histplot(numeric_df, x=columns, hue = 'CYLINDERS', palette = 'viridis')
#     plt.xticks(rotation=90)
fuelConsumption_df['CO2 EMISSIONS'].mean()


# ### The yearly trend of CO2 emissions btw 2010-2014

# In[368]:


sns.lineplot(x=fuelConsumption_df['MODELYEAR'], y=fuelConsumption_df['CO2 EMISSIONS'])
plt.show()


# ### Split Data into Train and Test sets

# In[80]:


#splitting our data  into 20% test and 80% train size
X,y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 100)
X,y = X_train, y_train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 100)


# In[79]:


X_train


# ### Linear Regression with all numeric data

# In[55]:


#scaling for each data group

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#X_val_scaled = scaler.transform(X_val)


# In[56]:


#converting to a dataframe

X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)
#X_val_scaled = pd.DataFrame(X_val_scaled, columns = X.columns)


# In[64]:


#instantiate the model

model = linear_model.LinearRegression()
model.fit(X_train, y_train)


# In[65]:


#training and testing datasets
### to check why values are not changing 


def report_model_test(model):
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    scores = mean_squared_error(y_test, predictions)
    absolute_mean = mean_absolute_error(y_test, predictions)
    print(f"The predictions are: {predictions}")
    print(f"The result of the mean squared error is : {np.sqrt(scores):.4f}")
    print(f"The result of the mean absolute error is : {absolute_mean:.4f}")


# In[60]:


report_model_test(model)


# ### Multiple linear regression using subset of numerical data

# In[70]:


#taking a subset of our numeric data for training

X1 = numeric_df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
       'FUELCONSUMPTION_HWY']]
y1 = numeric_df['CO2 EMISSIONS']
X, y =X_train, y_train
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state = 100)


# In[71]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#X_val_scaled = scaler.transform(X_val)


# In[72]:


#converting to a dataframe

X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)
#X_val_scaled = pd.DataFrame(X_val_scaled, columns = X.columns)


# In[73]:


#instantiate the model

model = linear_model.LinearRegression()
model.fit(X_train, y_train)


# In[74]:


report_model_test(model)


# ### Cross validation

# In[375]:


model = LogisticRegression()


# In[377]:


cv_result = cross_val_score(model,X_train,y_train, cv = 5,scoring = "accuracy")
cv_result


# In[378]:


cv_result.mean()


# ### Classification - to balance our dataset

# In[212]:


# fig, ax_position=plt.subplots(1,2,figsize=(15,6),dpi=200) # creates the framework for the plotting
sns.countplot(fuelConsumption_df['CO2 EMISSIONS'][:10]) 
plt.xticks(rotation=45)
plt.show()


# In[99]:


fuelConsumption_df['CO2 EMISSIONS'][:10].value_counts().plot.pie(autopct="%1.1f%%") #ax_position[1] specifies plot to be in index 1

plt.show()


# In[183]:


X = numeric_df.drop('CO2 EMISSIONS', axis = 1)
y = numeric_df['CO2 EMISSIONS']


# In[124]:


#calling our categorical variables df
objects_df


# ### Using Make as target variable

# In[195]:


#label encoding and creating df for model, to assign object variable to numbers
le = LabelEncoder()# declares an instance of the object
le_data = le.fit_transform(objects_df[['MAKE']])# applies the column modelyear to objectdata
df_le = pd.DataFrame(le_data, columns = ['MAKE']) # creates a dataframe
encode_make_df = pd.concat([numeric_df, df_le], axis=1)
encode_make_df.head()


# In[196]:


#plot of first 10rows after encoding
sns.countplot(encode_make_df['MAKE'][:10], data=encode_make_df) 
plt.show()


# ### Using Model as target variable

# In[197]:


le = LabelEncoder()# declares an instance of the object
le_data = le.fit_transform(objects_df[['MODEL']])# applies the column model to objectdata
df_le = pd.DataFrame(le_data, columns = ['MODEL']) # creates a dataframe
encodeModel_df = pd.concat([numeric_df, df_le], axis=1)
encodeModel_df.head()


# In[201]:


#plot of first 10rows after encoding
sns.countplot(encodeModel_df['MODEL'][:10], data=encodeModel_df) 
plt.xticks(rotation=45)
plt.show()


# ### using vehicleclass as target variable

# In[200]:


le = LabelEncoder()# declares an instance of the object
le_data = le.fit_transform(objects_df[['VEHCILECLASS']])# applies the column modelyear to objectdata
df_le = pd.DataFrame(le_data, columns = ['VEHCILECLASS']) # creates a dataframe
encodeVehicle_df = pd.concat([numeric_df, df_le], axis=1)
encodeVehicle_df.head()


# In[202]:


#plot of first 10rows after encoding
sns.countplot(encodeVehicle_df['VEHCILECLASS'][:10], data=encodeVehicle_df)
plt.xticks(rotation=45)
plt.show()


# ### using transmission as target variable

# In[203]:


le = LabelEncoder()# declares an instance of the object
le_data = le.fit_transform(objects_df[['TRANSMISSION']])# applies the column modelyear to objectdata
df_le = pd.DataFrame(le_data, columns = ['TRANSMISSION']) # creates a dataframe
encodeTransmission_df = pd.concat([numeric_df, df_le], axis=1)
encodeTransmission_df.head()


# In[204]:


#plot of first 10rows after encoding
sns.countplot(encodeTransmission_df['TRANSMISSION'][:10], data=encodeTransmission_df)
plt.xticks(rotation=45)
plt.show()


# ### using fuel type as target variable

# In[207]:


le = LabelEncoder()# declares an instance of the object
le_data = le.fit_transform(objects_df[['FUELTYPE']])# applies the column modelyear to objectdata
df_le = pd.DataFrame(le_data, columns = ['FUELTYPE']) # creates a dataframe
encodeFuelType_df = pd.concat([numeric_df, df_le], axis=1)
encodeFuelType_df.head()


# In[213]:


#plot of first 10rows after encoding
sns.countplot(encodeFuelType_df['FUELTYPE'][:10], data=encodeFuelType_df)
plt.xticks(rotation=45)
plt.show()


# ### To balance out target variable

# In[384]:


sns.countplot(encodeFuelType_df['FUELTYPE'][:10]) 
plt.xticks(rotation=45)
plt.show()


# In[262]:


encodeFuelType_df.dropna(inplace=True)
X = encodeFuelType_df.drop('FUELTYPE', axis=1)
y = encodeFuelType_df['FUELTYPE']


# In[265]:


objects_df


# ### Applying SMOTE TO FUELTYPE AND VEHICLECLASS

# In[242]:


#pip install -U imbalanced-learn
# installed this to be able to import the libraries


# In[259]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, k_neighbors = 2) # The object is created
X_res, y_res = sm.fit_resample(X, y) # The object is applied
X, y = X_res, y_res # reassigning the balanced dataset to X,y


# In[415]:


# Plot of the FUELTYPE object
balanced_df = pd.concat([X_res,y_res], axis = 1) # creating a dataframe for the balanced data
fig, ax=plt.subplots(1,2,figsize=(15,6)) # creating the axis shell for subplot
a = sns.countplot(x='FUELTYPE',data=balanced_df, ax=ax[0]) # assigning each of the plot to the axis shell
a= balanced_df['FUELTYPE'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax[1]) # assigning each of the plot to the axis shell


# In[416]:


def report_model_test(model):
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test)
    plt.show()
    plt.figure(figsize=(40,40), dpi = 200);


# In[417]:


model = RandomForestClassifier()
report_model_test(model)


# In[269]:


#to balance transmission

encodeVehicle_df.dropna(inplace=True)
X1 = encodeVehicle_df.drop('VEHCILECLASS', axis=1)
y1 = encodeVehicle_df['VEHCILECLASS']


# In[270]:


sm = SMOTE(random_state=42, k_neighbors = 2) # The object is created
X_res1, y_res1 = sm.fit_resample(X1, y1) # The object is applied
X1, y1 = X_res1, y_res1 # reassigning the balanced dataset to X,y


# In[281]:


# Plot of the Transmission object
balancedV_df = pd.concat([X_res1,y_res1], axis = 1) # creating a dataframe for the balanced data
fig, ax=plt.subplots(1,2,figsize=(15,6)) # creating the axis shell for subplot
a = sns.countplot(x='VEHCILECLASS',data=balancedV_df, ax=ax[0]) # assigning each of the plot to the axis shell
#plt.xticks(rotation=90)
a= balancedV_df['VEHCILECLASS'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax[1]) # assigning each of the plot to the axis shell
plt.xticks(rotation=90)


# ### Training, prediction and evaluation of model using Random forest

# In[387]:


#splitting our data  into 20% test and 80% train size

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 100)
X,y = X_train, y_train
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 100)


# In[399]:


#scaling the dataset
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_val_scaled = scaler.transform(X_val)


# In[389]:


X_train.shape 


# In[413]:


def report_model_test(model):
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test)
    plt.show()
    plt.figure(figsize=(40,40), dpi = 200);


# In[414]:


model = RandomForestClassifier()
report_model_test(model)


# In[393]:


model.summary()


# ### Clustering

# In[428]:


# numeric_df
# objects_df


# In[429]:


# ploting the attributes to see how they affect the clusters formed 
for col1 in numeric_df.columns:
     for col2 in numeric_df.columns:
        plt.figure(figsize = (8,9),dpi = 200)
        sns.set(font_scale=2)
        sns.scatterplot(data = numeric_df, x = numeric_df[col1],y = numeric_df[col2],
        hue='CYLINDERS', palette = 'viridis')
        plt.show()


# In[430]:


numeric_df.drop('MODELYEAR', axis = 1, inplace = True) # dropping model year since is not useful


# ### scaling data

# In[431]:


from sklearn.preprocessing import StandardScaler # using StandardScaler as variables are distance based, they need to be standardised


# In[432]:


# initialise the StandardScaler() class 
scaler = StandardScaler() 


# In[434]:


scaled_numeric_df=scaler.fit_transform(numeric_df)
scaled_numeric_df


# ### KMeans clustering

# In[435]:


# import KMeans 
from sklearn.cluster import KMeans 


# In[436]:


# assigning the number of clusters to 3
model = KMeans(n_clusters=3) 


# In[437]:


cluster_labels = model.fit_predict(scaled_numeric_df) # fiting and predicting the clusters
cluster_labels


# In[439]:


numeric_df['CLUSTER'] = cluster_labels


# In[440]:


numeric_df


# In[441]:


cluster_labels


# In[442]:


# ploting the attributes to see how they affect the clusters formed 
numeric_df.drop('CLUSTER', axis = 1, inplace = True)
for col1 in numeric_df.columns:
     for col2 in numeric_df.columns:
        plt.figure(figsize = (8,9),dpi = 200)
        sns.set(font_scale=2)
        sns.scatterplot(data = numeric_df, x = numeric_df[col1],y = numeric_df[col2],
        hue=cluster_labels, palette = 'viridis')
        plt.show()


# ### clustering evaluation

# In[445]:


from sklearn.metrics import davies_bouldin_score, silhouette_score


# In[447]:


DB_index=davies_bouldin_score(scaled_numeric_df,cluster_labels)
DB_index


# In[449]:


sil_score=silhouette_score(scaled_numeric_df,cluster_labels)
sil_score


# In[ ]:




