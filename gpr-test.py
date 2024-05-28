from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_excel ,DataFrame
from math import sqrt

# Load NIR reflectance data (X) and oil acidity data (y) 
db=read_excel("data-oil-no-mirrors.xlsx")#data-oil-2miroirs 
X=db.drop(['Unnamed: 0','Y'],axis=1)
Y=db['Y']
i=0
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Initialize GPR model with RBF kernel
gpr = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=10)

# Train model
gpr.fit(X_train, y_train)

# Make predictions
y_pred = gpr.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R2 : ", r2)
