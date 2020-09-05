from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("predictData.csv")  #Read in the data of 6 features 

#View data information again 
count_nan = data.isnull().sum() 
count_nan
# Read in data features and class labels
X = data.iloc[0:,0:6]#The features are  PARA_A  PARA_B  TOTAL  numbers  Money_Value  District in turn

#Standardized treatment
s_scaler = StandardScaler()
X = s_scaler.fit_transform(X.astype(np.float))

# Call the stacking model
StackingModel = joblib.load('sclf.pkl')
#Output the prediction results of the company's fraud
N = len(X)   #N is the number of companies
pred_label=[]
for i in range(N):
    pred_label.append(StackingModel.predict(X)[i])
    if StackingModel.predict(X)[i]==1:
        print("The %d company is suspected of fraud risk\n"%(i+1))
    else:
        print("The %d company has no suspected fraud risk\n"%(i+1))
data['predict label']=np.transpose(pred_label)
writer=pd.ExcelWriter("result_predict.xlsx")
data.to_excel(writer,sheet_name="predict")
writer.save()