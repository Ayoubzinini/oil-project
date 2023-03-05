def choosen_wl_filter(choosen_wl_file,X):
    from pandas import read_excel
    cwl=read_excel(choosen_wl_file)['choozen wavelengths values']
    return X.loc[:,cwl]
def ic_pr(x,y,model):
  from numpy import mean,sqrt,array,std,transpose,matmul,linalg
  from sklearn.metrics import mean_squared_error
  from scipy.stats import t
  n=x.shape[0]
  ich,ici=[],[]
  for i,j in zip(x.index,y):
      xh=x.loc[i,:]
      xh=[list(xh)]
      pr=model.predict(xh)[0]
      mse=mean_squared_error([j],[pr])
      xh=x.loc[i,:]
      xm=mean(xh)
      stderr=std(y-pr)*sqrt(abs(matmul(matmul(transpose(xh),linalg.inv(matmul(transpose(x),x))),xh)))
      #"""
      T=t(df=n-2).ppf(0.975)
      a=T*stderr
      ici.append(pr-a)
      ich.append(pr+a)
      #"""
  return DataFrame({'ICI':ici,'ICH':ich})