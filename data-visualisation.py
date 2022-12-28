from importlib.resources import path
from operator import index
from pandas import read_csv, read_excel
from matplotlib.pyplot import plot, show, xlabel, ylabel, legend
from statsmodels.multivariate.pca import PCA
from pandas.core.groupby.groupby import DataFrame
import statsmodels.api as sm
import pylab as py
import seaborn as sns
data=input('File name : ')
db=read_excel(data+".xlsx")
X=db.drop([db.columns[0],'Y'],axis=1)
Y=db['Y']

"""
for i in range(len(X.index)):
  plt.plot(X.columns,X.loc[X.index[i],:])
plt.xlabel("Wavelength")
plt.ylabel("R")
plt.title(data)
plt.show()
sns.heatmap(X.corr())
plt.show()
#"""
#"""
pc = PCA(X, ncomp=X.shape[0], method='nipals')
for i in pc.loadings.columns:
    plot(pc.loadings.index,pc.loadings[i])
    xlabel('WL')
    ylabel(i)
    show()
#"""