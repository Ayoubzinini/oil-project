from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, max_error, r2_score, mean_absolute_error
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from pandas import read_excel, DataFrame
import numpy as np
from scipy.stats import f_oneway, shapiro
from preproc_NIR import devise_bande
from random import randint
db=read_excel('data-oil-2miroirs.xlsx')
X=db.drop(['Unnamed: 0','Y'],axis=1)
Y=db['Y']
#"""
R2C,R2CV,R2T,RMSEC,RMSECV,RMSET,RDS=[],[],[],[],[],[],[]
j=0
while True:
    il=devise_bande(X,n_intervals=68)
    print(len(il))
    """
    if len(R2C) == len(R2CV) == len(R2T) == len(RMSEC) == len(RMSECV) == len(RMSET) == 0:
        idx=randint(0,len(il)-1)
        coord=il[idx]
        X=X[X.columns[coord[0]:coord[1]]]
        x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
        model=KNeighborsRegressor(algorithm='ball_tree',n_neighbors=3,weights='distance')
        model.fit(x_train,y_train)
        r2cv=r2_score(y_train,cross_val_predict(model,x_train,y_train,cv=LeaveOneOut()))
        r2c=r2_score(y_train,model.predict(x_train))
        r2t=r2_score(y_test,model.predict(x_test))
        rmsecv=mean_squared_error(y_train,cross_val_predict(model,x_train,y_train,cv=LeaveOneOut()))
        rmsec=mean_squared_error(y_train,model.predict(x_train))
        rmset=mean_squared_error(y_test,model.predict(x_test))
        if r2cv>0 and r2t>0:
            R2C.append(r2c)
            R2CV.append(r2cv)
            R2T.append(r2t)
            RMSEC.append(rmsec)
            RMSECV.append(rmsecv)
            RMSET.append(rmset)
            RDS.append(j)
    elif (len(R2C) == len(R2CV) == len(R2T) == len(RMSEC) == len(RMSECV) == len(RMSET) != 0) and len(R2C)<6:
        if len(il)==1:
            coord=il[0]
        else:
            idx=randint(0,len(il)-1)
            coord=il[idx]
        X=X[X.columns[coord[0]:coord[1]]]
        x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
        model=KNeighborsRegressor(algorithm='ball_tree',n_neighbors=3,weights='distance')
        model.fit(x_train,y_train)
        r2cv=r2_score(y_train,cross_val_predict(model,x_train,y_train,cv=LeaveOneOut()))
        r2c=r2_score(y_train,model.predict(x_train))
        r2t=r2_score(y_test,model.predict(x_test))
        rmsecv=mean_squared_error(y_train,cross_val_predict(model,x_train,y_train,cv=LeaveOneOut()))
        rmsec=mean_squared_error(y_train,model.predict(x_train))
        rmset=mean_squared_error(y_test,model.predict(x_test))
        if r2cv>0 and r2t>0:
            R2C.append(r2c)
            R2CV.append(r2cv)
            R2T.append(r2t)
            RMSEC.append(rmsec)
            RMSECV.append(rmsecv)
            RMSET.append(rmset)
            RDS.append(j)
    elif len(R2C) == len(R2CV) == len(R2T) == len(RMSEC) == len(RMSECV) == len(RMSET) == 6:
        break
    j+=1
    #"""
    break
#"""

results=DataFrame({"R²c":R2C,"R²CV":R2CV,"R²t":R2T,"RMSEc":RMSEC,"RMSEcv":RMSECV,"RMSEt":RMSET,"RDS":RDS})
print(results)
print(np.mean(results,axis=0))