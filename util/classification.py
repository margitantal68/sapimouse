import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics, model_selection

from sklearn.model_selection import cross_val_predict



# num_folds cross validation
# num_folds - number of folds
# df -dataframe containing class id in the last column

def evaluate_identification_CV(df, num_folds=3):
    print("CV identification")
    print(df.shape)
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]
    
    model = RandomForestClassifier(n_estimators=100)   
    scoring = ['accuracy']
    scores = cross_val_score(model , X ,y , cv = num_folds)
    for i in range(0,num_folds):
        print('\tFold '+str(i+1)+':' + str(scores[ i ]))    
    print("accuracy : %0.4f (%0.4f)" % (scores.mean() , scores.std()))  

    # y_pred = cross_val_predict(model, X, y, cv=3)
    # conf_mat = confusion_matrix(y, y_pred)
    # print(conf_mat)


# df_train -dataframe for training containing class id in the last column
# df_test  -dataframe for testing containing class id in the last column
def evaluate_identification_Train_Test(df_train, df_test):
    # Train data
    array = df_train.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X_train = array[:, 0:nfeatures]
    y_train = array[:, -1]
    
    # Test data
    array = df_test.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X_test = array[:, 0:nfeatures]
    y_test = array[:, -1]

    model = RandomForestClassifier(n_estimators=100)   
    model.fit(X_train, y_train)
    scoring = ['accuracy']
    score_value = model.score(X_test, y_test)
    print("accuracy : %0.4f" % (score_value)) 
    



