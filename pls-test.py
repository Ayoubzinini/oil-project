from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import normalize
import seaborn as sns
from pandas import read_csv, read_excel
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
import numpy as np
from scipy.stats import f_oneway
from scipy.signal import savgol_filter, detrend
from preproc_NIR import osc, msc, snv, simple_moving_average
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
X=DataFrame(savgol_filter(DataFrame(msc(X.to_numpy())),3,1,1))
choozen_idx=read_excel("choozen_wavelengths.xlsx")["choozen wavelengths index"]
X=X[choozen_idx]
Y=db['Y']
rescols=["r2c","r2cv","r2t","rmsec","rmsecv","rmset","rds"]
r2c,r2cv,r2t,rmsec,rmsecv,rmset,rds=[],[],[],[],[],[],[]
#111,237👌
j=0
while True:
  x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
  income_groups=[y_train,y_test]
  s,p=f_oneway(*income_groups)
  r2=[]
  RMSE=[]
  for i in range(1,31,1):
    model=PLSRegression(scale=False,n_components=i)
    model.fit(x_train, y_train)
    r2.append(r2_score(model.predict(x_test),y_test))
    RMSE.append(mean_squared_error(model.predict(x_test),y_test))
  model=PLSRegression(scale=False,n_components=1+RMSE.index(min(RMSE)))
  model.fit(x_train, y_train)
  RMSECV=abs(np.mean(cross_val_score(PLSRegression(n_components=1+RMSE.index(min(RMSE))), x_train, y_train, scoring='neg_root_mean_squared_error', cv=LeaveOneOut())))
  R2CV=100*np.mean(cross_val_score(PLSRegression(n_components=1+RMSE.index(min(RMSE))), x_train, y_train, scoring='r2'))
  R2train=100*r2_score(y_train,model.predict(x_train))
  R2test=100*r2_score(y_test,model.predict(x_test))
  RMSEtrain=mean_squared_error(y_train,model.predict(x_train))
  RMSEtest=mean_squared_error(y_test,model.predict(x_test))
  if R2CV>0 and R2test>0:
      """
      r2c.append(R2train)
      r2cv.append(R2CV)
      r2t.append(R2test)
      rmsec.append(RMSEtrain)
      rmsecv.append(RMSECV)
      rmset.append(RMSEtest)
      rds.append(j)
      res=DataFrame({rescols[0]:r2c,rescols[1]:r2cv,rescols[2]:r2t,rescols[3]:rmsec,rescols[4]:rmsecv,rescols[5]:rmset,rescols[6]:rds})
      if res.shape[0]==100:
          res.to_excel("res.xlsx")
          break
      #"""
      break
  j=j+1
print("pc nbr : ",1+RMSE.index(min(RMSE)))
print('Best random stat : ',j)
print(DataFrame([R2test,RMSEtest,R2train,RMSEtrain,RMSECV,R2CV],index=["R2test","RMSEtest","R2train","RMSEtrain","RMSECV","R2CV"],columns=["values"]))