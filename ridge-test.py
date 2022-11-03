import pandas as pd
import numpy as np
from matplotlib.pyplot import show, bar
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from pandas import read_csv, read_excel, DataFrame, concat
from scipy.stats import f_oneway
from pca import pca
def simple_moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
r=DataFrame({"one":np.ones(X.shape[1])})
for i in X.index:
  r=concat([r,DataFrame({str(i):simple_moving_average(X.loc[i,], window=5)})],axis=1)
X=r.T.drop("one",axis=0)
#X=X[X.columns[range(624,1024)]]
Y=db['Y']
while True:
  model = pca(n_components=20)
  results = model.fit_transform(X)
  x_train, x_test, y_train, y_test = train_test_split(results['PC'],Y,test_size=0.2)
  income_groups=[y_train,y_test]
  s,p=f_oneway(*income_groups)
  #parameters = {'alpha':np.logspace(-4, -3.5, 50)}  
  #ridge= GridSearchCV(linear_model.Ridge(), parameters, scoring='r2', cv=LeaveOneOut())
  ridge=linear_model.Ridge(alpha=0.0001)
  ridge.fit(x_train, y_train)
  #ridge1 = linear_model.Ridge(alpha=ridge.best_params_['alpha'])
  #Y_cv = cross_val_predict(ridge1, x_train, y_train, cv=LeaveOneOut())
  Y_cv = cross_val_predict(ridge, x_train, y_train, cv=LeaveOneOut())
  score_cv = r2_score(y_train, Y_cv)
  mse_cv = mean_squared_error(y_train, Y_cv)
  mse_train=mean_squared_error(y_train, ridge.predict(x_train))
  mse_test=mean_squared_error(y_test, ridge.predict(x_test))
  score_test=r2_score(y_test,ridge.predict(x_test))
  score_train=r2_score(y_train,ridge.predict(x_train))
  if p<0.05 and score_test>0 and score_cv>0:
    break
print('R2 CV (Ridge): ',100 * score_cv," %")
print('RMSE CV (Ridge): ',np.sqrt(mse_cv))
#print('Best parameter alpha = ', ridge.best_params_['alpha'])
print('R2 calibration: ',100 * score_train," %")
print('RMSE calibration: ',np.sqrt(mse_train))
print('R2 test: ',100 * score_test," %")
print('RMSE test: ',np.sqrt(mse_test))
bar([int(x) for x in range(len(ridge.coef_))], [i/np.sum(ridge.coef_**2) for i in ridge.coef_])
show()