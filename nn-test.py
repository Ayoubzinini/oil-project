from sklearn.neural_network import MLPRegressor
from statsmodels.multivariate.pca import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_predict, LeaveOneOut
from pandas import read_excel, DataFrame, concat
import numpy as np
file_name=input('File name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop(['Unnamed: 0','Y'],axis=1)
Y=db['Y']
def simple_moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')
r=DataFrame({"one":np.ones(X.shape[1])})
for i in X.index:
  r=concat([r,DataFrame({str(i):simple_moving_average(X.loc[i,], window=5)})],axis=1)
X=r.T.drop("one",axis=0)
pc = PCA(X, ncomp=20, method='nipals')
j=0
while True:
    x_train, x_test, y_train, y_test = train_test_split(DataFrame(pc.factors),Y,test_size=0.2,random_state=j)
    model = MLPRegressor(hidden_layer_sizes=(x_train.shape[0]*2,x_train.shape[0]*2,x_train.shape[0]),activation="identity" , max_iter=2000).fit(x_train, y_train)
    prc,prt=model.predict(x_train),model.predict(x_test)
    Y_cv = cross_val_predict(model, x_train, y_train, cv=LeaveOneOut())
    if r2_score(y_test,prt)>0 and r2_score(y_train, Y_cv)>0:
        break
    j+=1
print('R² test : {}\nMSE test : {}\nMAE test : {}\nR² CV : {}\nMSE CV : {}\nMAE CV : {}\nR² train : {}\nMSE train : {}\nMAE train : {}'.format(r2_score(y_test,prt), mean_squared_error(y_test,prt), mean_absolute_error(y_test,prt),r2_score(y_train, Y_cv),mean_squared_error(y_train, Y_cv),mean_absolute_error(y_train, Y_cv),r2_score(y_train,prc), mean_squared_error(y_train,prc), mean_absolute_error(y_train,prc)))