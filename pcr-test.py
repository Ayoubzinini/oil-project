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
def msc(input_data, reference=None):
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
    if reference is None:
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]
    return (data_msc, ref)
def snv(input_data):
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return output_data
def simple_moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
#X=X[X.columns[range(624,1024)]]
Y=db['Y']
while True:
  r=DataFrame({"one":np.ones(X.shape[1])})
  for i in X.index:
    r=concat([r,DataFrame({str(i):simple_moving_average(X.loc[i,], window=5)})],axis=1)
  X=r.T.drop("one",axis=0)
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