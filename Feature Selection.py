#Import lib and data
import joblib
import pandas as pd
import numpy as np
dat = pd.read_csv("trial.csv")
dat.head(3)
#Missing value check
count_nan = dat.isnull().sum() 
count_nan
#Fill in missing values
dat['Money_Value'].fillna(dat['Money_Value'].mean(), inplace = True)
dat.info()
#Divide X and Y
X,Y = dat.iloc[0:,0:9],dat.iloc[0:,9:]
#standardization
from sklearn import preprocessing
zscore = preprocessing.StandardScaler()
X = zscore.fit_transform(X)
#spearman Correlation coefficient for feature selection
dfs = dat.corr('spearman')  #Calculate spearman correlation coefficient
print(dfs)

dfs["Ranking"] = dfs["Risk"].rank(method="first")
display(dfs)#Print all spearman coefficient values

dfs['sort_helper'] = dfs['Risk'].abs()
dfs["absRanking"] = dfs["sort_helper"].rank(method="first")
display(dfs["absRanking"] )  #Ascending order

#RFE for feature selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 2)
rfe = rfe.fit(X,Y.astype('int'))
print(rfe.support_)
print(rfe.ranking_)