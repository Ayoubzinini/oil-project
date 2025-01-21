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
  from math import log10
  from pandas import DataFrame
  lg_X=DataFrame()
  for i in X.columns:
    lg_X[i]=[-log10(1/j) for j in X[i]]
  return lg_X
def devise_bande(X,n_intervals):
  Test=int(len(X.columns)/n_intervals)-(len(X.columns)/n_intervals)
  if Test>0:
    n_elements=2+int(len(X.columns)/n_intervals)
  elif Test<=0:
    n_elements=1+int(len(X.columns)/n_intervals)
  stop_vals=[]
  i=0
  while True:
    try:
      stop_vals.append(X.columns[i])
    except IndexError:
      stop_vals.append(X.columns[len(X.columns)-1])
      break
    i=i+n_elements
  stop_vals_=stop_vals[1:len(stop_vals)]
  intervals = [i for i in zip(stop_vals,stop_vals_)]
  x,y=[],[]
  for i in intervals:
    x.append(list(X.columns).index(i[0]))
    y.append(list(X.columns).index(i[1])+1)
  final_interval=[i for i in zip(x,y)]
  return final_interval
def var_sel(X,min_max_intervals_list):
  selected=[]
  for i in min_max_intervals_list:
    for j in X.columns:
      j=float(j)
      if j>=i[0] and j<=i[1]:
        selected.append(j)
  return X[selected]
def pow_trans(y,power_t):
  """
  This function transforms the training output of your model to reduce the noise of destribution and get a better training quality if it works well
  """
  from numpy import log
  #defining geometric mean function
  def gm(l):
    result=1
    for i in l:
      result *= i
    return result**(1/len(l))
  trans_y=[] #empty transformed vector
  if power_t==0:
    for i in y:
      trans_y.append(gm(y)*log(i)) #appendenig the transformed elements to the vector if power coefficent is null
  else:
    for i in y:
      trans_y.append((i**(power_t-1))/(power_t*gm(y)**(power_t-1)))  #appendenig the transformed elements to the vector if power coefficent not null
  return trans_y #full transformed vector
def ic_pr(x,y,model,risk=0.05):
  """"
  This function returns the prediction confident intervals of model's prediction
  You'll need : X input, Y output, model and risk of decision (5% by default) as arguments
  """
  from numpy import mean,sqrt,array,std,transpose,matmul,linalg
  from sklearn.metrics import mean_squared_error
  from scipy.stats import t
  n=x.shape[0]
  ich,ici=[],[] #ici : the down boundary of confident interval ; ich : the upper boundary of confident interval
  for i,j in zip(x.index,y):
      xh=x.loc[i,:] #sample selection
      xh=[list(xh)]
      pr=model.predict(xh)[0] #get prediction
      mse=mean_squared_error([j],[pr]) #get mean squared error
      xh=x.loc[i,:]
      stderr=sqrt(abs(mse*matmul(matmul(transpose(xh),linalg.inv(matmul(transpose(x),x))),xh))) #get standard error
      T=t(df=n-2).ppf(1-risk/2) #get student quantile that presents risk of decision
      a=T*stderr #get interval semi amplitude
      ici.append(pr-a) #get down boundary values
      ich.append(pr+a) #get upper boundary values
  return DataFrame({'ICI':ici,'ICH':ich})
def continuum_removal(wavelength, spectrum, window_size=10):
    """
    Continuum removal preprocessing for spectral data.
    
    Parameters:
    - wavelength: 1D array-like, the wavelength values.
    - spectrum: 1D array-like, the spectral intensity values.
    - window_size: int, the size of the window for local linear regression.
    
    Returns:
    - continuum_removed: 1D array, the continuum-removed spectrum.
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    continuum_removed = np.zeros_like(spectrum)
    for i in range(len(spectrum)):
        # Define local window
        start = max(0, i - window_size // 2)
        end = min(len(spectrum), i + window_size // 2)
        
        # Fit linear regression
        lr = LinearRegression()
        lr.fit(wavelength[start:end].reshape(-1, 1), spectrum[start:end])
        
        # Predict continuum value
        continuum = lr.predict(wavelength[i].reshape(-1, 1))
        
        # Remove continuum
        continuum_removed[i] = spectrum[i] - continuum
        
    return continuum_removed
def Hotellings_T2(data,n_components=2):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import f
    
    
    # Step 1: Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Step 2: Perform PCA
    pca = PCA()
    data_pca = pca.fit_transform(data_scaled)
    
    # Step 3: Select the number of PCs (e.g., 2 PCs)
    pca_scores = data_pca[:, :n_components]
    cov_matrix = np.cov(pca_scores, rowvar=False)
    
    # Step 4: Compute T² statistic for each sample
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    t2_stats = np.array([z.T @ inv_cov_matrix @ z for z in pca_scores])
    
    # Calculate threshold
    n_samples = data.shape[0]
    alpha = 0.05
    critical_value = f.ppf(1 - alpha, dfn=n_components, dfd=n_samples - n_components)
    threshold = (n_components * (n_samples - 1)) / (n_samples - n_components) * critical_value
    
    # Identify outliers
    outliers = np.where(t2_stats > threshold)[0]
    
    return outliers