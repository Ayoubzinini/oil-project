from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import normalize, StandardScaler
import seaborn as sns
from pandas import read_csv, read_excel
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
import numpy as np
from scipy.stats import f_oneway
from scipy.signal import savgol_filter, detrend
from preproc_NIR import osc, msc, snv, simple_moving_average, centring, prep_log
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
      stderr=sqrt(abs(mse*matmul(matmul(transpose(xh),linalg.inv(matmul(transpose(x),x))),xh)))
      #"""
      T=t(df=n-2).ppf(0.975)
      a=T*stderr
      ici.append(pr-a)
      ich.append(pr+a)
      #"""
  return DataFrame({'ICI':ici,'ICH':ich})
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
Y=db['Y']
Y=np.sqrt(Y)

rescols=["r2c","r2cv","r2t","rmsec","rmsecv","rmset","rds"]
r2c,r2cv,r2t,rmsec,rmsecv,rmset,rds=[],[],[],[],[],[],[]
for p_X in [osc(X),snv(X.iloc[:,:-1].values),msc(X.iloc[:,:-1].values),simple_moving_average(X),savgol_filter(X,5,1,1),normalize(X,axis=1),centring(X)]:
    p_X=DataFrame(p_X)
    j=0
    while True:
        x_train, x_test, y_train, y_test = train_test_split(p_X,Y,test_size=0.2,random_state=j)
        income_groups=[y_train,y_test]
        s,p=f_oneway(*income_groups)
        r2=[]
        RMSE=[]
        for i in range(1,30,1):
            model=PLSRegression(n_components=i)
            model.fit(x_train, y_train)
            r2.append(r2_score(model.predict(x_test),y_test))
            RMSE.append(mean_squared_error(model.predict(x_test),y_test))
        model=PLSRegression(n_components=1+RMSE.index(min(RMSE)))
        model.fit(x_train, y_train)
        RMSECV=abs(np.mean(cross_val_score(PLSRegression(n_components=1+RMSE.index(min(RMSE))), x_train, y_train, scoring='neg_root_mean_squared_error', cv=LeaveOneOut())))
        R2CV=100*np.mean(cross_val_score(PLSRegression(n_components=1+RMSE.index(min(RMSE))), x_train, y_train, scoring='r2'))
        R2train=100*r2_score(y_train,model.predict(x_train))
        R2test=100*r2_score(y_test,model.predict(x_test))
        RMSEtrain=mean_squared_error(y_train,model.predict(x_train))
        RMSEtest=mean_squared_error(y_test,model.predict(x_test))
        if p>0.05 and R2CV>0 and R2test>0:
            r2c.append(R2train)
            r2cv.append(R2CV)
            r2t.append(R2test)
            rmsec.append(RMSEtrain)
            rmsecv.append(RMSECV)
            rmset.append(RMSEtest)
            rds.append(j)
            break
        j=j+1
res=DataFrame({rescols[0]:r2c,rescols[1]:r2cv,rescols[2]:r2t,rescols[3]:rmsec,rescols[4]:rmsecv,rescols[5]:rmset,rescols[6]:rds})
print(res)
print(np.mean(res,axis=1))