from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from pandas import read_csv, read_excel
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
import numpy as np
from scipy.stats import f_oneway
def simple_moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
#X=X[X.columns[range(624,1024)]]
X.index=[str(i) for i in X.index]
r=DataFrame({"one":np.ones(X.shape[1])})
for i in X.index:
  r=concat([r,DataFrame({str(i):simple_moving_average(X.loc[i,], window=5)})],axis=1)
X=r.T.drop("one",axis=0)
Y=db['Y']
while True:
  x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
  income_groups=[y_train,y_test]
  s,p=f_oneway(*income_groups)
  r2=[]
  RMSE=[]
  for i in range(1,31,1):
    model=PLSRegression(n_components=i)
    model.fit(x_train, y_train)
    r2.append(model.score(x_test,y_test))
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
    break
print("pc nbr : ",1+RMSE.index(min(RMSE)))
print(DataFrame([R2test,RMSEtest,R2train,RMSEtrain,RMSECV,R2CV],index=["R2test","RMSEtest","R2train","RMSEtrain","RMSECV","R2CV"],columns=["values"]))