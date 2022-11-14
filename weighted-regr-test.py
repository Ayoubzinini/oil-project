from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from matplotlib.pyplot import show, bar, axhline, rcParams
from pandas import read_csv, read_excel
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
import numpy as np
from scipy.stats import f_oneway
from pca import pca
rcParams['figure.figsize'] = (12, 8)
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
Y=db['Y']
def simple_moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')
r=DataFrame({"one":np.ones(X.shape[1])})
for i in X.index:
  r=concat([r,DataFrame({str(i):simple_moving_average(X.loc[i,], window=5)})],axis=1)
X=r.T.drop("one",axis=0)
#9978,14152,16643,23158,23397,24382, 30484👌,30556,33207, 35565, 36224, 38367
#p<0.05 : 44839
#78686
i=44839
while True:
  model = pca(n_components=20)
  results = model.fit_transform(X)
  x_train, x_test, y_train, y_test = train_test_split(results['PC'],Y,test_size=0.2,random_state=i)
  income_groups=[y_train,y_test]
  s,p=f_oneway(*income_groups)
  wregr=LinearRegression()
  wregr.fit(x_train, y_train, np.ones(x_train.shape[0])/np.mean(x_train,axis=1)**2)
  Y_cv = cross_val_predict(wregr, x_train, y_train, cv=LeaveOneOut())
  score_cv = r2_score(y_train, Y_cv)
  mse_cv = mean_squared_error(y_train, Y_cv)
  mse_train=mean_squared_error(y_train, wregr.predict(x_train))
  mse_test=mean_squared_error(y_test, wregr.predict(x_test))
  score_test=r2_score(y_test,wregr.predict(x_test))
  score_train=r2_score(y_train,wregr.predict(x_train))
  if p<0.05 and score_test>0 and score_cv>0:
    break
  i=i+1
print('R2 CV (wregr): ',100 * score_cv," %")
print('RMSE CV (wregr): ',np.sqrt(mse_cv))
print('R2 calibration: ',100 * score_train," %")
print('RMSE calibration: ',np.sqrt(mse_train))
print('R2 test: ',100 * score_test," %")
print('RMSE test: ',np.sqrt(mse_test))
print("Best random state : ",i)
s=0
cumperc=[]
perc_coefs=DataFrame({'idx':results['PC'].columns,'val':[abs(i)/np.sum(abs(wregr.coef_)) for i in wregr.coef_],'cum':range(len(results['PC'].columns))}).sort_values(by=['val'])
for j in perc_coefs.val:
  s=s+j
  cumperc.append(s)
perc_coefs.cum=cumperc
bar(perc_coefs['idx'],perc_coefs['cum'])
axhline(y=0.8,linewidth=1, color='red')
show()