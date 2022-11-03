from scipy.signal import detrend, savgol_filter
from pandas import read_csv, read_excel, DataFrame
db=read_excel('c:/Users/ayoub/Downloads/oil-data-dt.xlsx')
wl=range(9,17,2)
po=range(1,8)
for i in range(len(wl)):
    for j in po:
        savgol_dt_X=savgol_filter(db.drop([db.columns[0]],axis=1),polyorder=j, window_length=wl[i])
        savgol_dt_X=DataFrame(savgol_dt_X)
        savgol_dt_X.to_excel("oil-data-dt+savgol"+"-"+str(i)+"-"+str(j)+".xlsx")