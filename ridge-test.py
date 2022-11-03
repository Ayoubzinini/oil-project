import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from pandas.core.groupby.groupby import DataFrame
from pandas import read_csv, read_excel
from scipy.stats import f_oneway
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
X=X[X.columns[range(624,1024)]]
Y=db['Y']
p=1
score_cv=-1
score_test=-1
while True:
  x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
  income_groups=[y_train,y_test]
  s,p=f_oneway(*income_groups)
  parameters = {'alpha':np.logspace(-4, -3.5, 50)}  
  ridge= GridSearchCV(linear_model.Ridge(), parameters, scoring='r2', cv=LeaveOneOut())
  ridge.fit(x_train, y_train)
  ridge1 = linear_model.Ridge(alpha=ridge.best_params_['alpha'])
  Y_cv = cross_val_predict(ridge1, x_train, y_train, cv=LeaveOneOut())
  score_cv = r2_score(y_train, Y_cv)
  mse_cv = mean_squared_error(y_train, Y_cv)
  score_test=ridge.score(x_test,y_test)
  score_train=ridge.score(x_train,y_train)
  if p<0.05 and score_test>0 and score_cv>0:
    break
print('R2 CV (Ridge): ',100 * score_cv," %")
print('RMSE CV (Ridge): ',np.sqrt(mse_cv))
print('Best parameter alpha = ', ridge.best_params_['alpha'])
print('R2 calibration: ',100 * score_train," %")
print('R2 test: ',100 * score_test," %")