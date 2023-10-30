from pandas import read_excel, DataFrame, read_csv
from scipy.signal import savgol_filter
from preproc_NIR import msc
import pickle
from os import listdir, getcwd
from itertools import compress
ls=listdir(getcwd())
pls=pickle.load(open("mwpls-model-oil.pkl", "rb"))
spec=read_csv(list(compress(ls,[i.endswith(".Spectrum") for i in ls]))[0],sep="\t")
spec=spec["y_Axis:%Reflectance or Transmittance"]
specs=DataFrame({"1":spec,"2":spec}).T
specs=DataFrame(savgol_filter(DataFrame(msc(specs.to_numpy())),3,1,1))
f_spec=specs.loc[0,]
f_spec=f_spec[read_excel("choozen_wavelengths.xlsx")["choozen wavelengths index"]]
print(pls.predict([f_spec])[0][0]-22.003135358399675)