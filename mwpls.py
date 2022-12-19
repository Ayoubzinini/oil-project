from preproc_NIR import devise_bande, snv
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.signal import savgol_filter, detrend
from pandas import read_excel ,DataFrame
from matplotlib.pyplot import plot, show, xlabel, ylabel, title
from numpy import sqrt
db=read_excel("data-oil-2miroirs.xlsx")
X=db.drop(['Unnamed: 0','Y'],axis=1)
Y=db['Y']
msecv=[]
r2cv=[]
min_i=[]
max_i=[]
print(devise_bande(X,13))
for i in devise_bande(X,13):
  inp=X.drop(list(X.columns[i[0]:i[1]]),axis=1)
  inp=DataFrame(snv(inp.iloc[:,:-1].values))
  j=0
  while True:
    x_train, x_test, y_train, y_test = train_test_split(inp,Y,test_size=0.2,random_state=j)
    pls=PLSRegression()
    pls.fit(x_train,y_train)
    ycv = cross_val_predict(pls, x_train, y_train, cv=LeaveOneOut())
    if r2_score(y_train,ycv)>0 and r2_score(y_test,pls.predict(x_test))>0:
      break
    j+=1
  msecv.append(mean_squared_error(y_pred=ycv,y_true=y_train))
  r2cv.append(r2_score(y_pred=ycv,y_true=y_train))
  min_i.append(i[0])
  max_i.append(i[1])
plot([i[0] for i in devise_bande(X,13)],[sqrt(i) for i in msecv],'--x')
xlabel('Intervals')
ylabel('RMSE CV')
show()
plot([i[0] for i in devise_bande(X,13)],r2cv,'--x')
xlabel('Intervals')
ylabel('R² CV')
show()