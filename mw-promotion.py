from preproc_NIR import devise_bande, msc, pow_trans, prep_log
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_squared_log_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.signal import savgol_filter, detrend
from pandas import read_excel ,DataFrame
from matplotlib.pyplot import plot, show, xlabel, ylabel, title, rcParams
from numpy import sqrt, mean
import time
import numpy as np
rcParams['figure.figsize'] = (12, 8)
db=read_excel("data-oil-2miroirs.xlsx")
X=db.drop(['Unnamed: 0','Y'],axis=1)
wl=X.columns
X=DataFrame(savgol_filter(DataFrame(msc(X.to_numpy())),3,1,1))
Y=db['Y']
Y=[np.sqrt(i) for i in Y]
mse,r2, err=[],[],[]
for i in range(1,30):
    pls=PLSRegression(n_components=i)
    pls.fit(X, Y)
    mse.append(mean_squared_error(Y, cross_val_predict(PLSRegression(n_components=i), X, Y, cv=LeaveOneOut())))
    r2.append(r2_score(Y, cross_val_predict(PLSRegression(n_components=i), X, Y, cv=LeaveOneOut())))
    err.append([abs(j-k) for j,k in zip([i[0] for i in pls.predict(X)],Y)])
plot(range(1,30),mse)
xlabel("PC Number")
ylabel("MSE")
title("MSE evolution")
show()
plot(range(1,30),r2)
xlabel("PC Number")
ylabel("R²")
title("R² evolution")
show()
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
for i in range(1,30):
    pls=PLSRegression(n_components=i)
    pls.fit(x_train, y_train)
    plot(y_train,pls.predict(x_train),".")
    plot(y_test,pls.predict(x_test),".")
    plot(Y,Y,"-")
    xlabel("Real values")
    ylabel("Predicted values")
    title("Prediction destributionution using "+str(i)+" components")
    show()