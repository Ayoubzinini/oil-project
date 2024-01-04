import pickle
from pandas import DataFrame, read_excel
from scipy.signal import savgol_filter
from preproc_NIR import msc
db=read_excel("data-oil-2024.xlsx",index_col="Unnamed: 0")
real_Y=[0.25,0.84,1.09,0.69,0.89,1.32,0.91,0.7,0.24,0.8,0.22,0.52,0.22,0.48,0.21,0.84,0.9,0.29]
X=DataFrame(savgol_filter(DataFrame(msc(db.to_numpy())),3,1,1))[read_excel("choozen_wavelengths.xlsx")["choozen wavelengths index"]]
pls=pickle.load(open("mwpls-model-oil.pkl", "rb"))
myY=read_clipboard()
rmyY=myY[::-1]
rmyY.index=myY.index
results=DataFrame({"Predicted":[i[0] for i in pls.predict(X)],"Real":real_Y},index=db.index)
results=DataFrame({"Predicted":[i[0] for i in pls.predict(X)],"Real":real_Y,"Index":[abs(i-j) for i,j in zip(results.Predicted,results.Real)]},index=db.index)