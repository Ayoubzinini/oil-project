from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from pandas import DataFrame, read_excel, concat
import matplotlib.pyplot as plt
from pandas.core.groupby.groupby import DataFrame
import numpy as np
from scipy.stats import f_oneway
from pca import pca
from preproc_NIR import simple_moving_average, snv, msc, osc
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
X=osc(X)
#X=X[X.columns[range(624,1024)]]
Y=db['Y']
while True:
  model = pca(n_components=20)
  results = model.fit_transform(X)
  x_train, x_test, y_train, y_test = train_test_split(results['PC'],db['Y'],test_size=0.2)
  income_groups=[y_train,y_test]
  s,p=f_oneway(*income_groups)
  r2=[]
  RMSE=[]
  for i in range(1,x_train.shape[0],1):
    model = pca(n_components=20)
    results = model.fit_transform(X)
    pcr=LinearRegression()
    pcr.fit(results['PC'],db['Y'])
    r2.append(r2_score(db['Y'],pcr.predict(results['PC'])))
    RMSE.append(mean_squared_error(db['Y'],pcr.predict(results['PC'])))
  model = pca(n_components=1+RMSE.index(min(RMSE)))
  results = model.fit_transform(X)
  pcr=LinearRegression()
  pcr.fit(x_train,y_train)
  RMSECV=abs(np.mean(cross_val_score(LinearRegression(), x_train, y_train, scoring='neg_root_mean_squared_error', cv=LeaveOneOut())))
  R2CV=100*np.mean(cross_val_score(LinearRegression(), x_train, y_train, scoring='r2'))
  prtrain=pcr.predict(x_train)
  R2train=100*r2_score(y_train,prtrain)
  prtest=pcr.predict(x_test)
  R2test=100*r2_score(y_test,prtest)
  RMSEtrain=mean_squared_error(y_train,prtrain)
  RMSEtest=mean_squared_error(y_test,prtest)
  if p<0.05 and R2CV>0 and R2test>0:
    break
print(DataFrame([R2test,RMSEtest,R2train,RMSEtrain,RMSECV,R2CV],index=["R2test","RMSEtest","R2train","RMSEtrain","RMSECV","R2CV"],columns=["values"]))