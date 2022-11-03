from pca import pca
from pandas.core.groupby.groupby import DataFrame
import numpy as np
from pandas import read_csv, read_excel
from matplotlib.pyplot import scatter, show
file_name=input('File Name : ')
db=read_excel(file_name+'.xlsx')
X=db.drop([db.columns[0],'Y'],axis=1)
model = pca(alpha=0.05, detect_outliers=['ht2', 'spe'])
results = model.fit_transform(X,col_labels=X.columns)
outliers_idx=[]
for i in range(len(results['outliers']['y_bool'])):
  if results['outliers']['y_bool'][i]==True:
    outliers_idx.append(i)
db=db.drop(outliers_idx)
db.to_excel(file_name+"-no.xlsx")