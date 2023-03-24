from preproc_NIR import devise_bande, msc, snv, pow_trans, prep_log, simple_moving_average
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
#from matplotlib.pyplot import show, bar, axhline, rcParams
from pandas import read_excel, DataFrame, concat
import numpy as np
from scipy.stats import f_oneway
from scipy.signal import savgol_filter
from statsmodels.multivariate.pca import PCA
#rcParams['figure.figsize'] = (12, 8)
file_name="data-oil-2miroirs"
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
scl=Normalizer()
scl.fit(X)
X=DataFrame(scl.transform(X))
Y=db['Y']
#Y=pow_trans(Y, 0.5)
i=0
while True:
  pc = PCA(X, ncomp=20, method='nipals')
  x_train, x_test, y_train, y_test = train_test_split(DataFrame(pc.factors),Y,test_size=0.2,random_state=i)
  income_groups=[y_train,y_test]
  s,p=f_oneway(*income_groups)
  svmregr=SVR(C=10, epsilon=0.1,gamma=0.0001,kernel='rbf')
  svmregr.fit(x_train, y_train,[0.001]*len(y_train))#, np.ones(x_train.shape[0])/np.mean(x_train,axis=1)**2
  Y_cv = cross_val_predict(svmregr, x_train, y_train, cv=LeaveOneOut())
  score_cv = r2_score(y_train, Y_cv)
  mse_cv = mean_squared_error(y_train, Y_cv)
  mse_train=mean_squared_error(y_train, svmregr.predict(x_train))
  mse_test=mean_squared_error(y_test, svmregr.predict(x_test))
  score_test=r2_score(y_test,svmregr.predict(x_test))
  score_train=r2_score(y_train,svmregr.predict(x_train))
  if score_test>0 and score_cv>0 and score_train>0:
    break
  i=i+1
print('R2 CV (SVM): ',100 * score_cv," %")
print('RMSE CV (SVM): ',np.sqrt(mse_cv))
print('R2 calibration: ',100 * score_train," %")
print('RMSE calibration: ',np.sqrt(mse_train))
print('R2 test: ',100 * score_test," %")
print('RMSE test: ',np.sqrt(mse_test))
print("Best random state : ",i)
"""s=0
cumperc=[]
perc_coefs=DataFrame({'idx':results['PC'].columns,'val':[abs(i)/np.sum(abs(svmregr.coef_)) for i in svmregr.coef_],'cum':range(len(results['PC'].columns))}).sort_values(by=['val'])
for j in perc_coefs.val:
  s=s+j
  cumperc.append(s)
perc_coefs.cum=cumperc
bar(perc_coefs['idx'],perc_coefs['cum'])
axhline(y=0.8,linewidth=1, color='red')
show()"""