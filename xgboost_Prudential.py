import time
import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa, quadratic_weighted_kappa_with_matrices
import random
import matplotlib.pyplot as plt

#from https://www.kaggle.com/chenglongchen/prudential-life-insurance-assessment/prudential-seed-23/code
#This script has been released under the Apache 2.0 open source license. 

#this is the modified version, different from the website, which I adjusted to understand what is going on.

random.seed(123)
start_time = time.time()


def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)

def eval_wrapper_with_matrices(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa_with_matrices(yhat, y)

def get_params():
    
    params = {}
    params["objective"] = "reg:linear"
    #params["objective"] = "reg:logistic"     
    params["eta"] = 0.045
    params["min_child_weight"] = 50
    #params["min_child_weight"] = 70 #my own optimization
    params["subsample"] = 0.8
    #params["colsample_bytree"] = 0.7
    params["silent"] = 1
    #params["silent"] = 0
    params["max_depth"] = 7 #from a script 
    #params["max_depth"] = 9 #read
    plst = list(params.items())

    return plst
    
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score


def main():
    # global variables
    columns_to_drop = ['Id', 'Response']
    xgb_num_rounds = 500
    num_classes = 8

    print("Load the data using pandas")
    #all_data = pd.read_csv("/home/reinhold/data/ML/Prudential/input_data/Prudential_1000.csv", header=0) ### useful for testing
    #train = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_cleaned.csv", header=0)
    #test = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/test_Prudential_cleaned.csv", header=0)
    #train = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_stripped_array_variables.csv", header=0)
    #test = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/test_Prudential_stripped_array_variables.csv", header=0)
    #train = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_afterPCA.csv", header=0)
    #test = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/test_Prudential_afterPCA.csv", header=0)
    train = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_standardized.csv", header=0)
    test = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/test_Prudential_standardized.csv", header=0)
    #train = pd.read_csv("/home/reinhold/data/ML/Prudential/input_data/train_Prudential.csv", header=0)
    #test = pd.read_csv("/home/reinhold/data/ML/Prudential/input_data/test_Prudential.csv", header=0)

    #print(train['Response'].unique())
    #print(train[train['Response']>0].shape)
    # combine train and test
    all_data = train.append(test)

    print('Eliminate missing values')    
    # Use -1 for any others
    all_data.fillna(-1, inplace=True)


    ## create any new variables    
    #all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[1]
    #all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[2]
    
    ## factorize categorical variables
    #all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    #all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
    #all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

    print(all_data['Response'].unique())
    print(all_data[all_data['Response']>0].shape)

    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)


    # Provide split column ###ToDo: not quite clear, where this is used. Inside xgboost?
    all_data['Split'] = np.random.randint(5, size=all_data.shape[0])
    
    # split train and test
    train = all_data[all_data['Response']>0].copy()
    test = all_data[all_data['Response']<1].copy()

    print(train.shape)
    print(test.shape)
    
    # convert data to xgb data structure
    xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
    #xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), label=train['Response'].values)
    xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)    
    
    # get the parameters for xgboost
    plst = get_params()
    print(plst)      

    
    # train model
    #print(xgb.get_params())

    model = xgb.train(plst, xgtrain, xgb_num_rounds) 
    print("after training: %.3f seconds" % (time.time() - start_time))

    # get preds
    train_preds = model.predict(xgtrain, ntree_limit=model.best_iteration)
    for i,y in enumerate(train_preds):
        print(i, y)
        if i>20: break

    training_score, numerator, denominator = eval_wrapper_with_matrices(train_preds, train['Response'])
    print('Train score is:', training_score)
    test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)
    train_preds = np.clip(train_preds, -0.99, 8.99)
    test_preds = np.clip(test_preds, -0.99, 8.99)
    print("after preds: %.3f seconds" % (time.time() - start_time))

    print("numerator:")
    print(numerator)
    print("denominator:")
    print(denominator)


    # train offsets 
    offsets = np.ones(num_classes) * -0.5
    offset_train_preds = np.vstack((train_preds, train_preds, train['Response'].values))
    for j in range(num_classes):
        train_offset = lambda x: -apply_offset(offset_train_preds, x, j)
        offsets[j] = fmin_powell(train_offset, offsets[j])  

    print("after training offsets: %.3f seconds" % (time.time() - start_time))
    
    # apply offsets to test
    data = np.vstack((test_preds, test_preds, test['Response'].values))
    for j in range(num_classes):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

    final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

    preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
    preds_out = preds_out.set_index('Id')
    #output_buffer = '/home/reinhold/data/ML/Prudential/output_data/py_xgb_stripped_array_variables.csv'
    #output_buffer = '/home/reinhold/data/ML/Prudential/output_data/py_xgb_afterPCA_train_only.csv'
    output_buffer = '/home/reinhold/data/ML/Prudential/output_data/py_xgb_afterPCA_Feb12.csv'
    preds_out.to_csv(output_buffer)
    print("file created: ", output_buffer)

if __name__ == "__main__":
    main()
    print("end: %.3f seconds" % (time.time() - start_time))
