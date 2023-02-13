from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import normalize, StandardScaler
import seaborn as sns
from pandas import read_csv, read_excel
from matplotlib.pyplot import plot, show, xlabel, ylabel, title
from pandas import DataFrame, concat
import numpy as np
from scipy.stats import f_oneway, shapiro
from scipy.signal import savgol_filter, detrend
from preproc_NIR import osc, msc, snv, simple_moving_average, centring, var_sel, pow_trans
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.sort_values(by='Y').drop([db.columns[0],'Y'],axis=1)
X=snv(DataFrame(savgol_filter(X,9,1,1)).iloc[:,:-1].values)
Y=db.sort_values(by='Y')['Y']
#Y=np.sqrt(Y)
rescols=["r2c","r2cv","r2t","rmsec","rmsecv","rmset"]
r2c,r2cv,r2t,rmsec,rmsecv,rmset=[],[],[],[],[],[]
flc,flt=[True]*22+[False]*7,[False]*22+[True]*7
x_train, x_test, y_train, y_test = X[flc],X[flt],Y[flc],Y[flt]
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
r2c.append(R2train)
r2cv.append(R2CV)
r2t.append(R2test)
rmsec.append(RMSEtrain)
rmsecv.append(RMSECV)
rmset.append(RMSEtest)
res=DataFrame({rescols[0]:r2c,rescols[1]:r2cv,rescols[2]:r2t,rescols[3]:rmsec,rescols[4]:rmsecv,rescols[5]:rmset})
print(res)
w,p = shapiro([i-j for i,j in zip(y_test,model.predict(x_test))])
if p>0.05:
  desicion="Normal"
elif p<0.05:
  desicion="Not Normal"
print('Quantile shapiro : {}\npropability shapiro : {}\ndesicion : {}'.format(w,p,desicion))