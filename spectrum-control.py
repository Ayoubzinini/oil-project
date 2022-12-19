from pandas import read_excel, DataFrame
from matplotlib.pyplot import plot, show, rcParams, legend
from numpy import nan
path=input('File name : ')
d1=read_excel(path+'.xlsx')
#"""
#"""
mc=input('wm or nm : ')
if mc=='wm':
  C=['fb8500','bc6c25','e63946','dda15e','ffb703','ffafcc','606c38','cdb4db','ffafcc','ccd5ae','ff006e','7209b7','c1121f','b5e48c','6a040f','9d0208','40916c','da2c38','00f5d4','8f2d56','8ac926','c9184a','ff4d6d','a7c957','006d77','8a817c','d90429','7209b7','6f1d1b']
  X=d1.drop(['Unnamed: 0','Y'],axis=1)
  X.index=d1['Unnamed: 0']
  Y=d1['Y']
  Y.index=d1['Unnamed: 0']
  Y_dscr=DataFrame({'count':[list(d1.Y).count(i) for i in d1.Y.unique()],'value':d1.Y.unique()})
  recap=DataFrame()
  l=[]
  for j in Y_dscr[Y_dscr['count']>1].value:
    idx=[]
    for i in Y.index:
      if d1.Y[i]==j:
        idx.append(i)
    l.append(len(idx))
    if len(idx)<max(l):
      idx=idx+[nan]*(max(l)-len(idx))
    recap[str(j)]=idx
  rcParams['figure.figsize'] = (12, 8)
  for a,b in zip(range(len(recap.columns)),recap.columns):
    print(str(a)+' : '+str(b))
  cho_val=int(input('choose a value : '))
  for i,j in zip(X.index,C):
    if len(recap[recap.columns[cho_val]].dropna())==3:
      cols=['#8ecae6','#219ebc','#023047']
    elif len(recap[recap.columns[cho_val]].dropna())==2:
      cols=['#8ecae6','#219ebc']
    for d,e in zip(recap[recap.columns[cho_val]].dropna(),cols):
      plot(X.columns,X.loc[d,],c=e,label=recap.columns[cho_val]+'-'+str(d))
    plot(X.columns,X.loc[i,],c='#'+j,label=str(Y[i])+'-'+str(i))
    legend(loc='best')
    show()
    input('press 1 to continue : ')
#"""
"""
elif mc=='nm':
  C=['fb8500','bc6c25','e63946','dda15e','ffb703','ffafcc','606c38','cdb4db','ffafcc','ccd5ae','ff006e','7209b7','c1121f','b5e48c','6a040f','9d0208','40916c','da2c38','00f5d4','8f2d56','8ac926','c9184a','ff4d6d','a7c957','006d77','8a817c','d90429','7209b7','6f1d1b']
  X=d2.drop(['Unnamed: 0','Y'],axis=1)
  X.index=d2['Unnamed: 0']
  Y=d2['Y']
  Y.index=d2['Unnamed: 0']
  Y_dscr=DataFrame({'count':[list(d2.Y).count(i) for i in d2.Y.unique()],'value':d2.Y.unique()})
  recap=DataFrame()
  l=[]
  print(Y_dscr[Y_dscr['count']>1])
  for j in Y_dscr[Y_dscr['count']>1].value:
    idx=[]
    for i in Y.index:
      if d2.Y[i]==j:
        idx.append(i)
    l.append(len(idx))
    if len(idx)<max(l):
      idx=idx+[nan]*(max(l)-len(idx))
    recap[str(j)]=idx
  rcParams['figure.figsize'] = (12, 8)
  for a,b in zip(range(len(recap.columns)),recap.columns):
    print(str(a)+' : '+str(b))
  cho_val=int(input('choose a value : '))
  for i,j in zip(X.index,C):
    if len(recap[recap.columns[cho_val]].dropna())==3:
      cols=['#8ecae6','#219ebc','#023047']
    elif len(recap[recap.columns[cho_val]].dropna())==2:
      cols=['#8ecae6','#219ebc']
    for d,e in zip(recap[recap.columns[cho_val]].dropna(),cols):
      plot(X.columns,X.loc[d,],c=e,label=recap.columns[cho_val]+'-'+str(d))
    plot(X.columns,X.loc[i,],c='#'+j,label=str(Y[i])+'-'+str(i))
    legend(loc='best')
    show()
    input('press 1 to continue : ')
#"""
