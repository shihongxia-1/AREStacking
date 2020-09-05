Jinfeng Mao, Hongxia Shi, Xinchen Cui, Yuchang Cai, Dekang Liu, Guangchen Liu, Mei Song. AREStacking: an Audit Risk Evaluation Model Based on ensemble machine learning Stacking algorithm,submitting

Corresponding author: ytsongmei@163.com

---------------------
Introduciton of AREStacking
---------------------
The data comes from the UCI Machine Learning Resource Library at the University of California, Irvine(UCI) (http://archive.ics.uci.edu/ml/index.php).  Dr. Nishtha Hooda provide this dataset in UCI, including 776 companies, 9 characteristics: PARA_A,ARA_B, Total, Money_Value, Numbers, Sector_score, Loss, History, and Distric,label is Risk. We thank Dr. Nishtha Hooda and the dataset supporter, i.e. Ministry of Electronics and Information Technology (MEITY), Govt.of India.
We use the company's fraud risk data with labeled tags (fraud or non-fraud) to explore the classification stacking algorithm, which will improve the accuracy of risk identification.


--------------------- 
files in the package
---------------------
1.The source data is in trial.csv
2.The code for feature selection is in Feature Selection.py
3.The data after feature selection is in trial_6.csv
4.The user's prediction data is in predictData.csv
5.The model training code is in Model_training.py
6.The trained model is AREStacking.pkl
7.The model predictive application code is prediction_AREStacking6.py


-------------------------
Pre-installed packages
-------------------------
pandas,numpy,mlxtend
sklearn:preprocessing,RFE,LogisticRegression,joblib,train_test_split,StandardScaler,GridSearchCV,DecisionTreeClassifier,KNeighborsClassifier,GradientBoostingClassifier,SVC,MLPRegressor,MLPClassifier,roc_curve,auc,model_selection


---------------------------------------------------------------------------------
Method Of predictive application for audit work (only for prediction users)
---------------------------------------------------------------------------------
1.Open the "predictData.csv" file, and keep the title of the six columns i.e. PARA_A,PARA_B,TOTAL,numbers,Money_Value,District, and replace the data by your own data. Save them.
2.Run "prediction_Stacking6.py" file.
3.The final prediction results will be output to the "result_predict.xlsx" file.

-------------------------------------------------------------------------------------------------------------
Whole precudures for training,testing and prediction (for users who plan to train their own models)
-------------------------------------------------------------------------------------------------------------
Step1.Feature selection
Step1.1 Open the "predictData.csv" file, and keep the title of the ten columns i.e. Sector_score,PARA_A,PARA_B,TOTAL,numbers,Money_Value,District,Loss,History,Risk, and replace the data by your own data. Save them.
Step1.2.Run "Feature Selection.py" file select the top 6 features.Output these 6 features and their corresponding column labels to the "trial_6.csv"file.
Step2.Model training
Step2.1 Open the "trial_6.csv" file.
Step2.2 Choose DT, KNN, GBDT, SVM, NN, Stackin to predict the data, and make the net style parameters for the first five models.
Step2.3 Substitute the optimal parameters obtained from the net style parameters into the corresponding models respectively, and integrate the five models after adjusting the parameters into the Stacking model to train these models.
Step2.4 Perform a 50% cross-validation.Comparing the Accuracy, F1 Score, Precision, Recall and AUC indexes of each model, it is found that the forecasting ability of Stacking model is the most outstanding.
Step2.5 Save the trained Stacking model.
Step3 Run "prediction_Stacking6.py" file to make predictions

------------------------
The end