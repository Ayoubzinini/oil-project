from preproc_NIR import devise_bande, msc, snv, pow_trans, prep_log, simple_moving_average
from scipy.signal import savgol_filter
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from pandas import read_excel, DataFrame, concat
import numpy as np
from statsmodels.multivariate.pca import PCA
file_name="data-oil-2miroirs"
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
X=DataFrame(savgol_filter(X,3,1,1))
Y=db['Y']
Y=[np.sqrt(i) for i in Y]
svr=SVR()
params={
        'kernel':('linear', 'rbf','poly','sigmoid'),
        'C':list(np.arange(1,10,1)),
        'gamma':list(np.arange(1e-5,1e-3,1e-6)),
        'epsilon':list(np.arange(0.1,0.9,0.1))
        }
opt=GridSearchCV(svr, params)
opt.fit(X, Y)
print(opt.best_params_)
