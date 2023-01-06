from pandas import read_excel
from scipy.stats import shapiro
db=read_excel("data-oil-normality.xlsx")
vec=db['Y']
w,p = shapiro(vec)
if p>0.05:
    print('Normal')
elif p<0.05:
    print('Not normal')
print("Probability shapiro : ",p)