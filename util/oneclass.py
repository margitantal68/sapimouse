import pandas as pd
import numpy as np
import util.normalization
import matplotlib.pyplot as plt  

from util.utils import create_userids
from util.plot import plot_scores
from random import  uniform


from sklearn.svm import OneClassSVM
from sklearn import metrics
from util.settings import TEMP_NAME, SCORES, SCORE_NORMALIZATION
from util.settings import RepresentationType


def calculate_EER(y, scores):
    # Calculating EER
    fpr,tpr, _ = metrics.roc_curve(y,scores,pos_label=1)
    fnr = 1-tpr
    # EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr + EER_fnr)
   
    return EER 

def compute_AUC(positive_scores, negative_scores):
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def compute_AUC_EER(positive_scores, negative_scores):  
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    fnr = 1-tpr   
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    return roc_auc, EER, fpr, tpr


def evaluate_authentication_train_test( df_train, df_test, data_type, num_blocks, representation_type, verbose = False, roc_data = False, roc_data_filename = TEMP_NAME):
    print("Training: "+str(df_train.shape))
    print("Testing: "+str(df_test.shape))
    userids = create_userids( df_train )
    NUM_USERS = len(userids)
    auc_list = list()
    eer_list = list()
    global_positive_scores = list()
    global_negative_scores = list()
    for i in range(0,NUM_USERS):
        userid = userids[i]
        user_train_data = df_train.loc[ df_train.iloc[:, -1].isin([userid]) ]
        # Select data for training
        user_train_data = user_train_data.drop(user_train_data.columns[-1], axis=1)
        user_array = user_train_data.values
        # train_samples = user_array.shape[0]
        
        user_test_data = df_test.loc[ df_test.iloc[:, -1].isin([userid]) ]
        user_test_data = user_test_data.drop(user_test_data.columns[-1], axis=1)
        # test_samples = user_test_data.shape[0]

        other_users_data = df_test.loc[~df_test.iloc[:, -1].isin([userid])]
        other_users_data = other_users_data.drop(other_users_data.columns[-1], axis=1)
        # other_users_array = other_users_data.values   
        
        # if (verbose == True):
        # print(str(userid)+". #train_samples: "+str(train_samples)+"\t#positive test_samples: "+ str(test_samples))

        clf = OneClassSVM(gamma='scale')
        clf.fit(user_train_data)
 
        positive_scores = clf.score_samples(user_test_data)
        negative_scores =  clf.score_samples(other_users_data)   
        
        # Aggregating positive scores
        y_pred_positive = positive_scores
        for i in range(len(positive_scores) - num_blocks + 1):
            y_pred_positive[i] = np.average(y_pred_positive[i : i + num_blocks], axis=0)

        # Aggregating negative scores
        y_pred_negative = negative_scores
        for i in range(len(negative_scores) - num_blocks + 1):
            y_pred_negative[i] = np.average(y_pred_negative[i : i + num_blocks], axis=0)

        auc, eer,_,_ = compute_AUC_EER(y_pred_positive, y_pred_negative)

        if SCORE_NORMALIZATION == True:
            positive_scores, negative_scores = score_normalization(positive_scores, negative_scores)

        global_positive_scores.extend(positive_scores)
        global_negative_scores.extend(negative_scores)

        if  verbose == True:
            print(str(userid)+", "+ str(auc)+", "+str(eer) )
         
        auc_list.append(auc)
        eer_list.append(eer) 
    print("\nNumber of blocks: ", num_blocks)
    print('AUC  mean : %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )
    print('EER  mean:  %7.4f, std: %7.4f' % ( np.mean(eer_list), np.std(eer_list)) )
    
    print("#positives: "+str(len(global_positive_scores)))
    print("#negatives: "+str(len(global_negative_scores)))


    global_auc, global_eer, fpr, tpr = compute_AUC_EER(global_positive_scores, global_negative_scores)
    
    
    filename = 'output_png/scores_'+ str(data_type.value)+ '_' + str(representation_type.value)
    if SCORES == True:
        # ****************************************************************************************
        plot_scores(global_positive_scores, global_negative_scores, filename, title='Scores distribution')
        # ****************************************************************************************

    if( roc_data == True ):
        dict = {'FPR': fpr, 'TPR': tpr}
        df = pd.DataFrame(dict) 
        df.to_csv(roc_data_filename, index=False)
        
    print(data_type.value + " Global AUC: "+str(global_auc))
    print(data_type.value + " Global EER: "+str(global_eer))
    return auc_list, eer_list


def score_normalization(positive_scores, negative_scores):
    scores = [positive_scores, negative_scores ]
    scores_df = pd.DataFrame( scores )

    # ZSCORE normalization
    mean = scores_df.mean()
    std = scores_df.std()
    min_score =  mean - 2 * std
    max_score = mean + 2 * std

    # MIN_MAX normalization
    # min_score = scores_df.min()
    # max_score = scores_df.max()

    min_score = min_score[ 0 ]
    max_score = max_score[ 0 ]

    positive_scores = [(x - min_score)/ (max_score - min_score ) for x in positive_scores] 
    positive_scores = [(uniform(0.0, 0.05) if x < 0 else  x) for x in positive_scores ]
    positive_scores = [ ( uniform(0.95, 1.0) if x > 1 else  x) for x in positive_scores ]

    negative_scores = [(x - min_score)/ (max_score - min_score )for x in negative_scores] 
    negative_scores = [ uniform(0.0, 0.05) if x < 0 else  x for x in negative_scores ]
    negative_scores = [ uniform(0.95, 1.0) if x > 1 else  x for x in negative_scores ]
    return positive_scores, negative_scores




