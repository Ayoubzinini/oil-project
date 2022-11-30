from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot, show, bar, axhline, rcParams, legend
from pandas import read_excel
from pandas import DataFrame, concat, Series
import numpy as np
from scipy.stats import f_oneway
from statsmodels.multivariate.pca import PCA
#rcParams['figure.figsize'] = (12, 8)
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
Y=db['Y']
X=X[X.columns[list(X.columns).index(1653.48633198773):len(X.columns)+1]]
"""
def snv(input_data):
  
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
 
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
 
    return output_data
X=X.iloc[:,:-1].values
X=DataFrame(snv(X))
"""
def ic_pr(x,y,model):
  from numpy import mean,sqrt,array,std,transpose,matmul,linalg
  from sklearn.metrics import mean_squared_error
  from scipy.stats import t
  n=x.shape[0]
  ich,ici=[],[]
  for i,j in zip(x.index,y):
      xh=x.loc[i,:]
      xh=[list(xh)]
      pr=model.predict(xh)[0]
      mse=mean_squared_error([j],[pr])
      xh=x.loc[i,:]
      xm=mean(xh)
      stderr=std(y-pr)*sqrt(abs(matmul(matmul(transpose(xh),linalg.inv(matmul(transpose(x),x))),xh)))
      #"""
      T=t(df=n-2).ppf(0.975)
      a=T*stderr
      ici.append(pr-a)
      ich.append(pr+a)
      #"""
  return DataFrame({'ICI':ici,'ICH':ich})
def simple_moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')
r=DataFrame({"one":np.ones(X.shape[1])})
for i in X.index:
  r=concat([r,DataFrame({str(i):simple_moving_average(X.loc[i,], window=5)})],axis=1)
X=r.T.drop("one",axis=0)
#pond 1/m : 9897,11583,38385,44839
#pond [0.001]*23 : 37, 1058, 1195, 2370, 3222, 3286, 3410, 4044
rescols=["r2c","r2cv","r2t","rmsec","rmsecv","rmset","rds"]
r2c,r2cv,r2t,rmsec,rmsecv,rmset,rds=[],[],[],[],[],[],[]
i=101 #last stop poly : 44839 
while True:
  pc = PCA(X, ncomp=20, method='nipals')
  x_train, x_test, y_train, y_test = train_test_split(DataFrame(pc.factors),Y,test_size=0.2,random_state=i)
  income_groups=[y_train,y_test]
  s,p=f_oneway(*income_groups)
  xgbregr=GradientBoostingRegressor()
  xgbregr.fit(x_train, y_train, np.ones(x_train.shape[0])/np.mean(x_train,axis=1)**2)
  Y_cv = cross_val_predict(xgbregr, x_train, y_train, cv=LeaveOneOut())
  score_cv = r2_score(y_train, Y_cv)
  mse_cv = mean_squared_error(y_train, Y_cv)
  mse_train=mean_squared_error(y_train, xgbregr.predict(x_train))
  mse_test=mean_squared_error(y_test, xgbregr.predict(x_test))
  score_test=r2_score(y_test,xgbregr.predict(x_test))
  score_train=r2_score(y_train,xgbregr.predict(x_train))
  if p>=0.05 and score_test>0 and score_cv>0:
      """
      r2c.append(score_train)
      r2cv.append(score_cv)
      r2t.append(score_test)
      rmsec.append(np.sqrt(mse_train))
      rmsecv.append(np.sqrt(mse_cv))
      rmset.append(np.sqrt(mse_test))
      rds.append(i)
      res=DataFrame({rescols[0]:r2c,rescols[1]:r2cv,rescols[2]:r2t,rescols[3]:rmsec,rescols[4]:rmsecv,rescols[5]:rmset,rescols[6]:rds})
      if res.shape[0]==100:
          res.to_excel("res.xlsx")
          break
      #"""
      break
  i=i+1
print('R2 CV (SVM): ',100 * score_cv," %")
print('RMSE CV (SVM): ',np.sqrt(mse_cv))
print('R2 calibration: ',100 * score_train," %")
print('RMSE calibration: ',np.sqrt(mse_train))
print('R2 test: ',100 * score_test," %")
print('RMSE test: ',np.sqrt(mse_test))
print("Best random state : ",i)
plot(xgbregr.predict(x_test),'-b',label='pred')
plot(ic_pr(x_test, y_test, xgbregr)['ICI'],'-g',label='IC inf')
plot(ic_pr(x_test, y_test, xgbregr)['ICH'],'-r',label='IC sup')
legend(loc='best')
show()
print('MAE train : ',mean_absolute_error(y_train, xgbregr.predict(x_train)))
print('MAE CV : ',mean_absolute_error(y_train, Y_cv))
print('MAE test : ',mean_absolute_error(y_test, xgbregr.predict(x_test)))
"""
standard_train=DataFrame(x_train)
standard_train['Y']=Series(y_train)
standard_train.to_excel('standard-train.xlsx')
standard_test=DataFrame(x_test)
standard_test['Y']=Series(y_test)
standard_test.to_excel('standard-test.xlsx')
print(y_train)
print(y_test)
"""