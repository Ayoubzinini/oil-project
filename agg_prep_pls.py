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
X=X.drop(list(X.columns[0:79])+list(X.columns[795:811]),axis=1)#
Y=db['Y']
Y=np.sqrt(Y)
#Y=pow_trans(Y,1.5)
rescols=["r2c","r2cv","r2t","rmsec","rmsecv","rmset","rds"]
r2c,r2cv,r2t,rmsec,rmsecv,rmset,rds=[],[],[],[],[],[],[]

for p_X in [savgol_filter(DataFrame(msc(X.iloc[:,:-1].values)),13,1,1)]:
    p_X=DataFrame(p_X)
    j=1
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
        if R2CV>0 and R2test>0:
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
print(np.mean(res,axis=0))
w,p = shapiro([i-j for i,j in zip(y_test,model.predict(x_test))])
if p>0.05:
  desicion="Normal"
elif p<0.05:
  desicion="Not Normal"
coefs=[i[0] for i in model.coef_]
coefs.append((np.mean(y_train) - np.dot(np.mean(x_train),model.coef_)))
DataFrame({'C':coefs}).to_json("coefs_model_oil.json")
DataFrame(x_train).to_excel("Calibration_Data_oil.xlsx")
print('Quantile shapiro : {}\npropability shapiro : {}\ndesicion : {}'.format(w,p,desicion))
plot(cross_val_predict(model, x_train, y_train, cv=LeaveOneOut()),[i-j for i,j in zip(y_train,cross_val_predict(model, x_train, y_train, cv=LeaveOneOut()))],'.')
xlabel('y pr')
ylabel('e')
show()