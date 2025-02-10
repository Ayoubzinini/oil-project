import pickle
from pandas import read_excel
#importing data
data=read_excel("C:/Users/ayoub/Downloads/OneDrive_1_11-11-2024/data-oil-10-24-final.xlsx")
X=data.drop(['Unnamed: 0','Y'],axis=1)
Y=data["Y"]
choozen_idx=read_excel("choozen_wavelengths.xlsx")["choozen wavelengths values"]
#choosing best wavelengths
X=X[choozen_idx]
#choosing sample example
sample=X.loc[0,:]
#importing model
PLS=pickle.load(open("mwpls-model-oil.pkl", "rb"))
#predict oil acidity
prediction=PLS.predict([sample])[0][0]
#prediction error correction
if (prediction >= 1.9) & (prediction <= 3):
    final_prediction=prediction+2.331585
else:
    final_prediction=abs(prediction)
print("Oil acidity is : ",final_prediction)
#coefficients :
coefs=PLS.coef_
intercept=0.0172318361981536