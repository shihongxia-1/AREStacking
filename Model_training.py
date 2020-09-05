import pandas as pd
import numpy as np
data = pd.read_csv("trial_6.csv")  #Read in the data of 6 features 

# View data information 
count_nan = data.isnull().sum() 
count_nan

#Missing value filling 
data['Money_Value'].fillna(data['Money_Value'].mean(), inplace = True)
#View data information again 
count_nan = data.isnull().sum() 
count_nan

X,y = data.iloc[0:,0:6],data.iloc[0:,6]#There are 776 samples and each sample contains 9 features. 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Standardization (training set and test set)
s_scaler = StandardScaler()
X = s_scaler.fit_transform(X.astype(np.float))
#Divide training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33 ,random_state=1)

#Define a general function for printing training results 
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#Grid parameter adjustment 
#dt
dt = DecisionTreeClassifier()
parameters_dt = {'max_depth':range(2,20,1),'min_samples_split':range(50,801,50)}
cv_dt=GridSearchCV(dt,parameters_dt,cv=5) #Define training strategy of training model 
cv_dt.fit(X_train,y_train).score
print_results(cv_dt)  #Output training results 

pdt = [cv_dt.best_params_["max_depth"],cv_dt.best_params_["min_samples_split"]]
pdt.append(cv_dt.best_params_["max_depth"])
pdt.append(cv_dt.best_params_["min_samples_split"])

#knn
knn = KNeighborsClassifier()
parameters_knn = {
    'n_neighbors': [3,4,5,6,7,8,9,10,11,12,13,14,15],
    'weights': ['uniform','distance'],
    'algorithm':['auto','ball_tree','kd_tree','brute']
}

cv_knn=GridSearchCV(knn,parameters_knn,cv=5) #Define training strategy of training model 
cv_knn.fit(X_train,y_train) 
print_results(cv_knn)  #Output training results 


pknn = [cv_knn.best_params_["n_neighbors"],cv_knn.best_params_["weights"],cv_knn.best_params_["algorithm"]]

#gb
gb =  GradientBoostingClassifier()
parameters_gbdt1 = {'n_estimators':range(15,25,1),
              'subsample':[0.3,0.4,0.5,0.6,0.7,0.8,0.9]
             }
cv_gbdt=GridSearchCV(gb,parameters_gbdt1,cv=5) #Define training strategy of training model 
cv_gbdt.fit(X_train,y_train) 
print_results(cv_gbdt)  #Output training results 


pgb = [cv_gbdt.best_params_["n_estimators"],cv_gbdt.best_params_["subsample"]]
parameters_gbdt2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
cv_gbdt2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators = pgb[0], min_samples_leaf=20, subsample=pgb[1], random_state=10), 
   param_grid = parameters_gbdt2, scoring='roc_auc',iid=False, cv=5)
cv_gbdt2.fit(X_train,y_train)
print_results(cv_gbdt2)

pgb.append(cv_gbdt2.best_params_["max_depth"])
pgb.append(cv_gbdt2.best_params_["min_samples_split"])

#svc
svc = SVC(gamma='scale')  #Create training model 
parameters_svm = {
    'kernel': ['linear','poly','rbf','sigmod'],
    'C': [10,11,12,13,14]
}

cv_svm=GridSearchCV(svc,parameters_svm,cv=5) #Define training strategy of training model 
cv_svm.fit(X_train,y_train) 
print_results(cv_svm)  #Output training results

psvc = []
psvc.append(cv_svm.best_params_["kernel"])
psvc.append(cv_svm.best_params_["C"])

#nn
mlp = MLPClassifier()
parameters_nn = {
    'hidden_layer_sizes': [(22,),(23,),(24,),(25,),(26,)],
    'activation': ['tanh','identity','logistic','relu'],
    'learning_rate': ['adaptive','constant','invscaling',],
    'solver':['lbfgs','sgd','adam']
}
cv_nn=GridSearchCV(mlp,parameters_nn,cv=5)
cv_nn.fit(X_train,y_train)
print_results(cv_nn)

pnn = [cv_nn.best_params_["hidden_layer_sizes"],cv_nn.best_params_["activation"],
       cv_nn.best_params_["learning_rate"],cv_nn.best_params_["solver"]]

#Calculation model index 
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
dt = DecisionTreeClassifier(max_depth = pdt[0], min_samples_split = pdt[1])
knn = KNeighborsClassifier(n_neighbors= pknn[0], weights = pknn[1], algorithm = pknn[2])
gb =  GradientBoostingClassifier( n_estimators = pgb[0], subsample = pgb[1], max_depth = pgb[2], min_samples_split = pgb[3])
svc =  SVC( kernel = psvc[0], C = psvc[1])
nn = MLPClassifier(hidden_layer_sizes = pnn[0], activation= pnn[1], learning_rate = pnn[2], solver = pnn[3])
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[dt,knn,gb,svc,nn],meta_classifier=lr)
print('5-fold cross validation:\n')
for clf, label in zip([dt,knn,gb,svc,nn,sclf],                       
                      ['dt',
                       'knn',                                           
                       'gb',       
                       'svc',
                       'nn',
                       'StackingClassifier']):
    scoring=['accuracy','f1','precision','recall','roc_auc']
    
    acc = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (acc.mean(),acc.std(), label))
    
    ff = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
    print("F1: %0.3f (+/- %0.3f) [%s]" % (ff.mean(),ff.std(), label))
    
    pre = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring='precision')
    print("precision: %0.3f (+/- %0.3f) [%s]" % (pre.mean(),pre.std(), label))
    
    rcl = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring='recall')
    print("recall: %0.3f (+/- %0.3f) [%s]" % (rcl.mean(),rcl.std(), label))
    
    rac = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
    print("roc_auc: %0.3f (+/- %0.3f) [%s]" % (rac.mean(),rac.std(), label))

#Generate the trained stacking model
from sklearn.externals import joblib
sclf.fit(X_train,y_train)
pre_sclf = sclf.predict(X_train)
joblib.dump(sclf,'sclf.pkl')



 