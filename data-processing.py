from scipy.signal import detrend, savgol_filter
from pandas import read_csv, read_excel, DataFrame
db=read_excel('c:/Users/ayoub/Downloads/oil-data.xlsx')
dt_X=detrend(db.drop([db.columns[0],'AC'],axis=1))
dt_X=DataFrame(dt_X)
db['AC'].to_excel('oil-rep.xlsx')
dt_X.to_excel("oil-data-dt.xlsx")