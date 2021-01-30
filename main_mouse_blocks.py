import pandas as pd  
import numpy as np

from util.oneclass import  evaluate_authentication_train_test
from util.model import get_model_output_features, train_model
from util.normalization  import normalize_rows, normalize_columns, normalize_all

from util.classification import evaluate_identification_Train_Test
from util.settings import DataType, RepresentationType
import util.settings as st

from util.plot import plot_ROC_filelist, plot_ROC_single
from util.classification import evaluate_identification_CV
from util.utils import create_userids

from enum import Enum

class RawDataType( Enum ):
    vx_vy = "vx_vy"
    ABS_vx_vy = "ABS_vx_vy"
    dx_dy = "dx_dy"
    ABS_dx_dy = "ABS_dx_dy"


# feature learning
# used subjects: 1..72 (session 3 min + session 1 min)
def train_feature_extractor():
    df1 = pd.read_csv("input_csv_mouse/sapimouse_" + raw_data_type.value +"_3min.csv")
    df2 = pd.read_csv("input_csv_mouse/sapimouse_" + raw_data_type.value +"_1min.csv")
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    users_train = [x for x in range(1,73)]
    df1_train =  df1.loc[ df1.iloc[:, -1].isin([ x for x in users_train ]) ]
    df2_train =  df2.loc[ df2.iloc[:, -1].isin([ x for x in users_train ]) ]

    frames = [df1_train, df2_train]
    df_train = pd.concat(frames)
    train_model(df_train, model_name, num_filters, representation_learning=True)


# users_eval - subsetset of users used for evaluation (session 3 min + session 1 min)
# num_blocks - number of blocks used for score computations

def verification_block_traintest( users_eval, num_blocks = 1, verbose = False):
    df1 = pd.read_csv("input_csv_mouse/sapimouse_" + raw_data_type.value +"_3min.csv")
    df2 = pd.read_csv("input_csv_mouse/sapimouse_" + raw_data_type.value +"_1min.csv")
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    df1_eval =  df1.loc[ df1.iloc[:, -1].isin(users_eval) ]
    df2_eval =  df2.loc[ df2.iloc[:, -1].isin(users_eval) ]
  
    # Feature extraction
    df1_eval_features = get_model_output_features( df1_eval, model_name )
    df2_eval_features = get_model_output_features( df2_eval, model_name )

    # Authentication
    roc_data_filename2 = 'results/roc_' + str(num_blocks) + '.csv'
    evaluate_authentication_train_test( df1_eval_features, df2_eval_features, data_type, num_blocks, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename2)

    # plot ROC
    title = 'ROC curve ' + str(data_type.value) + '_' + str(raw_data_type.value)
    plot_ROC_single(roc_data_filename2, title = title)



# VERIFICATION
# input parameters:
#       num_blocks - number of blocks used for decision
# output: 
#       file containing positive and negatives scores
#       results/roc_[num_blocks].csv

def verification(num_blocks):
    users = range(73,121)
    verification_block_traintest(users, num_blocks = num_blocks, verbose = False)


# MAIN
if __name__ == "__main__":
    dataset_name ='sapimouse_72_'
    data_type = DataType.MOUSE
    # Select the type of raw features
    raw_data_type = RawDataType.ABS_dx_dy
    # number of FCN filters
    num_filters = 128
    representation_type = RepresentationType.EE
    model_name = dataset_name + '_fcn_' + str(num_filters) + '_' + str(raw_data_type.value) + '.h5'
    
    # Train feature extractor
    train_feature_extractor()
    
    # User authentication/verification
    for num_blocks in range(1,6):
        print('Number of blocks: ', num_blocks)
        verification(num_blocks)

    # ROC curves for blocks 1..5
    filelist = ['results/roc_1.csv', 'results/roc_2.csv', 'results/roc_3.csv', 'results/roc_4.csv', 'results/roc_5.csv']
    plot_ROC_filelist(filelist, title = 'Sapimouse - 48 users', outputfilename='output_png/roc_sapimouse_48.png')



# UNIT = block of 128 events
# def classification_block_traintest( users, raw_data_type = RawDataType.vx_vy ):
#     num_filters = 128
#     df1 = pd.read_csv("input_csv_mouse/sapimouse_" + raw_data_type.value +"_3min.csv")
#     df2 = pd.read_csv("input_csv_mouse/sapimouse_" + raw_data_type.value +"_1min.csv")
    
#     df1 =  df1.loc[ df1.iloc[:, -1].isin(users) ]
#     df2 =  df2.loc[ df2.iloc[:, -1].isin(users) ]

#     df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
#     df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

#     # EE features
#     model_name = dataset_name + '_fcn_' + str(num_filters) + '_' + str(raw_data_type.value) + '.h5'
#     # model_name = 'balabit_fcn_128.h5'
#     print(model_name)
#     df1_features = get_model_output_features( df1, model_name )
#     df2_features = get_model_output_features( df2, model_name )

#     evaluate_identification_Train_Test(df1_features, df2_features)

# # UNIT = block of 128 events
# def classification_block_train( users, raw_data_type = RawDataType.vx_vy ):
#     num_filters = 128
#     df1 = pd.read_csv("input_csv_mouse/sapimouse_" + raw_data_type.value +"_3min.csv")
#     df1 =  df1.loc[ df1.iloc[:, -1].isin(users) ]
    
#     df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    
#     # EE features
#     model_name = dataset_name + '_fcn_' + str(num_filters) + '_' + str(raw_data_type.value) + '.h5'
#     # model_name = 'balabit_fcn_128.h5'
#     df1_features = get_model_output_features( df1, model_name )
#     evaluate_identification_CV(df1_features)


# # UNIT = block of 128 events
# # classification based on raw data, without the feature learning step
# def classification_block_raw_traintest( users, raw_data_type = RawDataType.vx_vy ):
#     num_filters = 128
#     df1 = pd.read_csv("input_csv_mouse/sapimouse_" + raw_data_type.value +"_3min.csv")
#     df1 =  df1.loc[ df1.iloc[:, -1].isin(users) ]
#     df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)

#     df2 = pd.read_csv("input_csv_mouse/sapimouse_" + raw_data_type.value +"_1min.csv")
#     df2 =  df2.loc[ df2.iloc[:, -1].isin(users) ]
#     df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)
#     evaluate_identification_Train_Test(df1, df2)


# # CLASSIFICATION/IDENTIFICATION
# # feature learning: sapimouse users 1 .. 72
# # classification: Random Forests (100 trees)
# #                 training: 3 min session
# #                 test: 1 min session 
# def classification(raw_data_type):
#     users = range(72,121)
#     classification_block_traintest( users, raw_data_type= raw_data_type )