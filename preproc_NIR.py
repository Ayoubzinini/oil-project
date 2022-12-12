def osc(X):
  from pandas import read_excel, DataFrame
  from statsmodels.multivariate.pca import PCA
  from numpy import matmul, transpose
  recap=DataFrame()
  pc=PCA(X,method='nipals')
  for i in X.columns:
    a=X.loc[:,i]
    b=DataFrame(matmul(pc.scores,transpose(pc.loadings)))
    b.columns=X.columns
    b=b.loc[:,i]
    re=[]
    for j,k in zip(a,b):
      re.append(j-k)
    recap[i]=re
  return recap
def simple_moving_average(X, window=5):
  from numpy import convolve, ones
  from pandas import DataFrame, concat
  X.index=[str(i) for i in X.index]
  r=DataFrame({"one":ones(X.shape[1])})
  for i in X.index:
    signal=X.loc[i,]
    r=concat([r,DataFrame({str(i):convolve(signal, ones(window)/window, mode='same')})],axis=1)
  X_mv=r.T.drop("one",axis=0)
  return X_mv
def msc(input_data, reference=None):
    from pandas import DataFrame
    from numpy import mean, zeros_like, polyfit
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
    if reference is None:
        ref = mean(input_data, axis=0)
    else:
        ref = reference
    data_msc = zeros_like(input_data)
    for i in range(input_data.shape[0]):
        fit = polyfit(ref, input_data[i,:], 1, full=True)
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]
    return DataFrame(data_msc)
def snv(input_data):
    from pandas import DataFrame
    from numpy import mean, zeros_like, std
    output_data = zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i,:] = (input_data[i,:] - mean(input_data[i,:])) / std(input_data[i,:])
    return DataFrame(output_data)
def centring(X):
  from numpy import mean
  from pandas import DataFrame
  Xc=DataFrame()
  for i in X.columns:
    Xc[i]=X[i]-mean(X[i])
  return Xc
def prep_log(X):
  from numpy import log10
  from pandas import DataFrame
  lg_X=DataFrame()
  for i in X.columns:
    lg_X[i]=log10(1/X[i])
  return lg_X