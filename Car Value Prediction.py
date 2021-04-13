# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:50:51 2021

@author: ylin26
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 08:25:29 2021

@author: ylin26
"""
# Importing all required packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks
from sklearn.impute import KNNImputer
from statsmodels.graphics.factorplots import interaction_plot


df = pd.DataFrame(pd.read_csv(r'C:\Users\ylin26\Box\My Box\Cox_Interview\Interview_Data.csv'))
df.head()
df.shape
df.describe()
df.info()

item_counts = df["SUBSEGMENTNAME"].value_counts()
print(item_counts.shape)

sns.boxplot(y=df['SOLD PRICE'])
sns.distplot(df['SOLD PRICE'])

df = pd.DataFrame(pd.read_csv(r'C:\Users\ylin26\Box\My Box\Cox_Interview\Interview_Data_clean.csv'))
# Checking Null values
df.isnull().sum()
# Condition grade has 25 missing values in the dataset.
# missing values get imputed based on the KNN algorithm i.e. K-nearest-neighbour algorithm.
#Apply KNN imputation algorithm
imputer = KNNImputer(n_neighbors=3)
imputed = pd.DataFrame(imputer.fit_transform(df[['conditiongrade','odometerkm','modelyear']]))
df['conditiongrade']= round(imputed[[0]],0)

#Exploratory Data Analysis ( EDA )
#Car Sold Price
sns.distplot(df['soldprice'])


# look at miles by year boxplot
fig = plt.figure(figsize=(15,5))
ax = fig.gca()
sns.boxplot(x='odometerkm',y='modelyear',data=df,notch=True,orient='h')
plt.xlabel('odometerkm',fontsize=14,fontweight="bold")
ax.set_yticklabels(sorted(df.modelyear.unique()))
plt.ylabel('modelyear',fontsize=14,fontweight="bold")
plt.title('Distribution of Odometers per Model Year',fontsize=18,fontweight="bold")
plt.xlim(0,150000)
plt.show()

# look at miles by make boxplot
# limit upper prices so it's more informative

fig = plt.figure(figsize=(16,11))
ax = fig.gca()
sns.boxplot(x='soldprice',y='brandname',data=df,
            order=sorted(df.brandname.unique()),notch=True,orient='h')
plt.xlabel('Price ($)',fontsize=14,fontweight="bold")
plt.ylabel('Brand Name',fontsize=14,fontweight="bold")
plt.title('Price Distribution per Brand',fontsize=18,fontweight="bold")
plt.show()

# look at price by year boxplot
# limit upper prices so it's more informative
fig = plt.figure(figsize=(10,5))
ax = fig.gca()
sns.boxplot(x='soldprice',y='modelyear',data=df,
            order=sorted(df.modelyear.unique()),notch=True,orient='h')
plt.xlabel('Price ($)',fontsize=14,fontweight="bold")
#ax.set_yticklabels(sorted(cc_data.modelYear.unique()))
plt.ylabel('Model Year',fontsize=14,fontweight="bold")
plt.title('Price Distribution per Model Year',fontsize=18,fontweight="bold")
plt.show()


# look at price by make boxplot
# limit upper prices so it's more informative
fig = plt.figure(figsize=(16,11))
ax = fig.gca()
sns.boxplot(x='soldprice',y='brandname',data=df,
            order=sorted(df.brandname.unique()),notch=True,orient='h')
plt.xlabel('Price ($)',fontsize=14,fontweight="bold")
plt.ylabel('Brand Name',fontsize=14,fontweight="bold")
plt.title('Price Distribution per Make',fontsize=18,fontweight="bold")
plt.show()

fig = plt.figure(figsize=(16,11))
ax = fig.gca()
sns.boxplot(x='soldprice',y='conditiongrade',data=df,
            order=sorted(df.brandname.unique()),notch=True,orient='h')
plt.xlabel('Price ($)',fontsize=14,fontweight="bold")
plt.ylabel('Condition Grade',fontsize=14,fontweight="bold")
plt.show()

# Let's see companies and their no of models.
fig, ax = plt.subplots(figsize = (15,5))
plt1 = sns.countplot(df['brandname'])
plt1.set(xlabel = 'brandname', ylabel= 'Count of Cars')
xticks(rotation = 90)
plt.title('Brand Count',fontsize=18,fontweight="bold")
plt.show()
plt.tight_layout()

# Let's see companies and their no of models.
fig, ax = plt.subplots(figsize = (10,5))
plt1 = sns.countplot(df['country'])
plt1.set(xlabel = 'country', ylabel= 'Count of Cars')
xticks(rotation = 90)
plt.show()
plt.tight_layout()

#Fuel Type by sold Price;
fig = interaction_plot(x=df['class'],trace=df['country'],response=df['soldprice'],
            colors=['pink','blue','purple','green'], ms=10)

df_class_avg_price = df[['class','soldprice']].groupby("class", as_index = False).mean().rename(columns={'soldprice':'class_avg_price'})
df_country_avg_price = df[['country','soldprice']].groupby("country", as_index = False).mean().rename(columns={'soldprice':'country_avg_price'})
df_year_avg_price = df[['yr_diff','soldprice']].groupby("yr_diff", as_index = False).mean().rename(columns={'soldprice':'year_avg_price'})
plt1 = df_class_avg_price.plot(x = 'class', kind='bar',legend = False, sort_columns = True)
plt1 = df_country_avg_price.plot(x = 'country', kind='bar',legend = False, sort_columns = True)

plt1.set_xlabel("country")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
plt.show()

plt1 = df_year_avg_price.plot(x = 'yr_diff', kind='bar',legend = False, sort_columns = True)


fig = interaction_plot(x=df['soldyear'],trace=df['modelyear'],response=df['soldprice'],
            colors=['pink','blue','purple'], ms=10)
df_yrdiff_avg_price = df[['yr_diff','soldprice']].groupby("yr_diff", as_index = False).mean().rename(columns={'soldprice':'class_avg_price'})
df_class_avg_price = df[['class','soldprice']].groupby("class", as_index = False).mean().rename(columns={'soldprice':'class_avg_price'})
df_country_avg_price = df[['country','soldprice']].groupby("country", as_index = False).mean().rename(columns={'soldprice':'country_avg_price'})
plt1 = df_class_avg_price.plot(x = 'class', kind='bar',legend = False, sort_columns = True)
plt1 = df_country_avg_price.plot(x = 'country', kind='bar',legend = False, sort_columns = True)
plt1 = df_yrdiff_avg_price.plot(x = 'yr_diff', kind='bar',legend = False, sort_columns = True)

plt1.set_xlabel("Years before sold")
plt1.set_ylabel("Avg Price (Dollars)")
xticks(rotation = 0)
plt.show()

plt1 = sns.scatterplot(x = 'odometerkm', y = 'soldprice', hue = 'class', data = df)
plt1.set_xlabel('odometer')
plt1.set_ylabel('Price of Car (Dollars)')
plt.show()

plt1 = sns.scatterplot(x = 'odometerkm', y = 'soldprice', hue = 'conditiongrade', data = df)
plt1.set_xlabel('odometer')
plt1.set_ylabel('Price of Car')
plt.show()