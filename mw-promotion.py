from preproc_NIR import devise_bande, msc, pow_trans, prep_log
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
wl=X.columns
X=DataFrame(savgol_filter(DataFrame(msc(X.to_numpy())),3,1,1))
Y=db['Y']
Y=[np.sqrt(i) for i in Y]
mse,r2=[],[]
for i in range(1,30):
    pls=PLSRegression(n_components=i)
    mse.append(mean_squared_error(Y, cross_val_predict(pls, X, Y, cv=LeaveOneOut())))
    r2.append(r2_score(Y, cross_val_predict(pls, X, Y, cv=LeaveOneOut())))
#plot(range(1,30),mse)
plot(range(1,30),r2)
show()