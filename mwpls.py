from preproc_NIR import devise_bande, msc, pow_trans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.signal import savgol_filter, detrend
from pandas import read_excel ,DataFrame
from matplotlib.pyplot import plot, show, xlabel, ylabel, title
from numpy import sqrt, mean
import time
import numpy as np
db=read_excel("data-oil-2miroirs.xlsx")
X=db.drop(['Unnamed: 0','Y'],axis=1)
X=DataFrame(savgol_filter(X,3,1,1))
Y=db['Y']
"""
msecv=[]
r2cv=[]
min_i=[]
max_i=[]
#"""
j=0
while True:
  x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
  pls=PLSRegression()
  pls.fit(x_train,y_train)
  ycv = cross_val_predict(pls, x_train, y_train, cv=LeaveOneOut())
  if r2_score(y_train,ycv)>0 and r2_score(y_test,pls.predict(x_test))>0:
    break
  j+=1
best_msecv=mean_squared_error(y_train,ycv)
best_r2cv=r2_score(y_train,ycv)
best_r2c=r2_score(y_train,pls.predict(x_train))
best_r2t=r2_score(y_test,pls.predict(x_test))
bands=devise_bande(X,13)
while True:
  try:
      for i in bands:
        inp=X.drop(list(X.columns[i[0]:i[1]]),axis=1)
        j=0
        program_starts = time.time()
        while True:
          x_train, x_test, y_train, y_test = train_test_split(inp,Y,test_size=0.2,random_state=j)
          pls=PLSRegression()
          pls.fit(x_train,y_train)
          ycv = cross_val_predict(pls, x_train, y_train, cv=LeaveOneOut())
          now = time.time()
          run_time = now - program_starts
          if (r2_score(y_train,ycv)>0 and r2_score(y_test,pls.predict(x_test))>0):
            break
          if run_time>60:
              break
          j+=1
        if r2_score(y_train,pls.predict(x_train))>best_r2c and r2_score(y_test,pls.predict(x_test))>best_r2t and r2_score(y_pred=ycv,y_true=y_train)>best_r2cv:
          best_r2cv=r2_score(y_pred=ycv,y_true=y_train)
          best_r2c=r2_score(y_train,pls.predict(x_train))
          best_r2t=r2_score(y_test,pls.predict(x_test))
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
print(best_r2cv)
print(len(X.columns))
print("Test : ",r2_score(y_test,pls.predict(x_test)))
print("Train : ",r2_score(y_train,pls.predict(x_train)))
coefs=[i[0] for i in pls.coef_]
coefs.append((np.mean(y_train) - np.dot(np.mean(x_train),pls.coef_)))
DataFrame({'C':coefs}).to_json("coefs_model_oil.json")
DataFrame(x_test).to_excel("test_Data_oil.xlsx")
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