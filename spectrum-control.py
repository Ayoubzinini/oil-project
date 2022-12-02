from pandas import read_excel
from matplotlib.pyplot import plot, show, rcParams, legend

d1=read_excel('data-oil-2miroirs.xlsx')
d2=read_excel('data-oil-12+31janv.xlsx')
idx=[]
for i in range(len(d1.Y)):
  if d1.Y[i]==0.41:# or d1.Y[i]==0.3 or d1.Y[i]==0.24
    idx.append(i)

"""
X=d1.drop(['Unnamed: 0','Y'],axis=1)
from matplotlib.pyplot import plot, show, rcParams, legend
rcParams['figure.figsize'] = (12, 8)
for i in X.index:
  plot(X.columns,X.loc[3,],c='#8ecae6')
  plot(X.columns,X.loc[17,],c='#219ebc')
  plot(X.columns,X.loc[18,],c='#023047')
  plot(X.columns,X.loc[i,],c='#bf0603')
#legend(loc='best')
show()
#"""
#"""
C=['fb8500','bc6c25','e63946','dda15e','ffb703','ffafcc','606c38','cdb4db','ffafcc','ccd5ae','ff006e','7209b7','c1121f','b5e48c','6a040f','9d0208','40916c','da2c38','00f5d4','8f2d56','8ac926','c9184a','ff4d6d','f4f3ee','bcb8b1','8a817c']
X=d1.drop(['Unnamed: 0','Y'],axis=1)
X.index=d1['Unnamed: 0']
rcParams['figure.figsize'] = (12, 8)
for i,j in zip(X.index,C):
  plot(X.columns,X.loc['145-13',],c='#8ecae6',label='0.41-145-13')
  plot(X.columns,X.loc['C12-17',],c='#219ebc',label='0.41-C12-17')
  plot(X.columns,X.loc['C12-9',],c='#023047',label='0.41-C12-9')
  plot(X.columns,X.loc[i,],c='#'+j,label=i)
  legend(loc='best')
  show()
  input('press 1 to continue : ')
#"""
