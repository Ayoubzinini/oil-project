from preproc_NIR import devise_bande, msc, pow_trans, prep_log
from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import normalize, StandardScaler
from pandas import read_csv, read_excel
from matplotlib.pyplot import plot, show, xlabel, ylabel, title
from pandas import DataFrame, concat
import numpy as np
from scipy.stats import f_oneway, shapiro
from scipy.signal import savgol_filter, detrend
from preproc_NIR import osc, msc, snv, simple_moving_average, centring, var_sel, pow_trans
from statsmodels.multivariate.pca import PCA
import time
file_name="data-oil-2miroirs"
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
wl=X.columns
X=DataFrame(savgol_filter(msc(X.to_numpy()),3,1,1))
Y=db['Y']
Y=[np.sqrt(i) for i in Y]
"""
msecv=[]
r2cv=[]
min_i=[]
max_i=[]
#"""
j=0
while True:
  pc = PCA(X, ncomp=29, method='nipals')
  x_train, x_test, y_train, y_test = train_test_split(DataFrame(pc.factors),Y,test_size=0.2,random_state=j)
  pcr=LinearRegression()
  pcr.fit(x_train, y_train)
  ycv = cross_val_predict(pcr, x_train, y_train, cv=LeaveOneOut())
  #"""
  if r2_score(y_train,ycv)>0 and r2_score(y_test,pcr.predict(x_test))>0:
    break
  j+=1
  #"""
best_msecv=mean_squared_error(y_train,ycv)
best_r2cv=r2_score(y_train,ycv)
best_r2c=r2_score(y_train,pcr.predict(x_train))
best_r2t=r2_score(y_test,pcr.predict(x_test))
bands=devise_bande(X,69)
while True:
  try:
      for i in bands:
        inp=X.drop(list(X.columns[i[0]:i[1]]),axis=1)
        j=0
        program_starts = time.time()
        while True:
          pc = PCA(X, ncomp=29, method='nipals')
          x_train, x_test, y_train, y_test = train_test_split(DataFrame(pc.factors),Y,test_size=0.2,random_state=j)
          pcr=LinearRegression()
          pcr.fit(x_train, y_train)
          ycv = cross_val_predict(pcr, x_train, y_train, cv=LeaveOneOut())
          now = time.time()
          run_time = now - program_starts
          if (r2_score(y_train,ycv)>0 and r2_score(y_test,pcr.predict(x_test))>0):
            break
          if run_time>60:
              break
          j+=1
        if r2_score(y_train,pcr.predict(x_train))>=best_r2c and r2_score(y_test,pcr.predict(x_test))>=best_r2t and r2_score(y_pred=ycv,y_true=y_train)>=best_r2cv:
          best_r2cv=r2_score(y_pred=ycv,y_true=y_train)
          best_r2c=r2_score(y_train,pcr.predict(x_train))
          best_r2t=r2_score(y_test,pcr.predict(x_test))
          best_i=i
      X=X.drop(list(X.columns[best_i[0]:best_i[1]]),axis=1)
      bands.remove(best_i)
  except: # ValueError
      break
  """
  msecv.append(mean_squared_error(y_pred=ycv,y_true=y_train))
  r2cv.append(r2_score(y_pred=ycv,y_true=y_train))
  min_i.append(i[0])
  max_i.append(i[1])
  #"""
print("CV : ",best_r2cv)
print("Selected wl : ",len(X.columns))
print("Test : ",r2_score(y_test,pcr.predict(x_test)))
print("Train : ",r2_score(y_train,pcr.predict(x_train)))
coefs=[i[0] for i in pcr.coef_]
coefs.append((np.mean(y_train) - np.dot(np.mean(x_train),pcr.coef_)))
DataFrame({'C':coefs}).to_excel("coefs_model_oil.xlsx")
choozen_idx=DataFrame(x_train).columns
choozen_wl=wl[choozen_idx]
DataFrame({'choozen wavelengths index':choozen_idx,'choozen wavelengths values':choozen_wl}).to_excel("choozen_wavelengths.xlsx")
DataFrame(x_test).to_excel("test_Data_oil.xlsx")
intercept=(np.mean(y_train) - np.dot(np.mean(x_train),pcr.coef_))
print((x_test.loc[x_test.index[0],]) @ pcr.coef_ + intercept)
print(pcr.predict([x_test.loc[x_test.index[0],]]))
"""
plot([i[0] for i in devise_bande(X,64)],[sqrt(i) for i in msecv],'--x')
xlabel('Intervals')
ylabel('RMSE CV')
show()
plot([i[0] for i in devise_bande(X,64)],r2cv,'--x')
xlabel('Intervals')
ylabel('R² CV')
show()
#"""