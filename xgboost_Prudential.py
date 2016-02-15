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
    #params["min_child_weight"] = 50
    params["min_child_weight"] = 70 #my own optimization
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    #params["silent"] = 0
    #params["max_depth"] = 7 #from a script 
    params["max_depth"] = 9 #read
    plst = list(params.items())

    return plst
    
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score


def main():
    minmax = [(1,2),(3,4),(5,6),(7,8)]
    for i in minmax:
        loop(i)


def loop(minmax_):


    path =  ('/home/reinhold/data/ML/Prudential/intermediate_data/',
             '/home/reinhold/data/ML/Prudential/output_data/')

    train = { 'in': path[0] + 'train_Prudential_pred_afterPCA_resp%d-%d.csv' % (minmax_[0], minmax_[1]),
              'out': path[1] + 'train_Prudential_pred_BDT_resp%d-%d.csv' % (minmax_[0], minmax_[1])}
    test = { 'in': path[0] + 'test_Prudential_pred_afterPCA_resp%d-%d.csv' % (minmax_[0], minmax_[1]),
             'out': path[1] + 'test_Prudential_pred_BDT_resp%d-%d.csv' % (minmax_[0], minmax_[1])}

    train_labels = { 'in': path[0] + "train_Prudential_predicted_labels_pred_resp%d-%d.csv" % (minmax_[0], minmax_[1]),
                     'out': path[1] + "train_Prudential_predicted_labels_BDT_pred_resp%d-%d.csv" % (minmax_[0], minmax_[1])}

    test_labels = { 'in': path[0] + "test_Prudential_predicted_labels_pred_resp%d-%d.csv" % (minmax_[0], minmax_[1]),
                     'out': path[1] + "test_Prudential_predicted_labels_BDT_pred_resp%d-%d.csv" % (minmax_[0], minmax_[1])}

    filenames = [train, train_labels, test, test_labels] 

    print("Load the data using pandas")
    #all_data = pd.read_csv("/home/reinhold/data/ML/Prudential/input_data/Prudential_1000.csv", header=0)
    df = []
    df[len(df):] = [pd.read_csv(train['in'], header=0)]
    df[len(df):] = [pd.read_csv(train_labels['in'], header=0)]
    df[len(df):] = [pd.read_csv(test['in'], header=0)]
    df[len(df):] = [pd.read_csv(test_labels['in'], header=0)] 
    
    for i in df:
        print(i.shape)

    df_train_labels = df[1]
    #include the neigboring responses:
    unique_responses = df_train_labels['Response'].unique()
    if min(unique_responses)<1: print("unexpected min response value: %d" % min(unique_responses))
    if max(unique_responses)>8: print("unexpected max response value: %d" % max(unique_responses))

    print(minmax_)

    # global variables
    columns_to_drop = ['Id', 'Response']
    xgb_num_rounds = 500
    num_classes = len(unique_responses) #can be modified below

    print('Eliminate missing values')    
    # Use -1 for any others
    for i in df:
        i.fillna(-1, inplace=True)
        print(i.shape)
    
    
    # convert data to xgb data structure
    xgtrain = xgb.DMatrix(df[0].drop(columns_to_drop, axis=1), df[1]['Response'].values) #training set
    #xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), label=train['Response'].values)
    xgtest = xgb.DMatrix(df[2].drop(columns_to_drop, axis=1), label=df[3]['Response'].values) #test set    
    
    # get the parameters for xgboost
    plst = get_params()
    print(plst)      

    model = xgb.train(plst, xgtrain, xgb_num_rounds) 
    print("after training: %.3f seconds" % (time.time() - start_time))

    # get preds
    train_preds = model.predict(xgtrain, ntree_limit=model.best_iteration)
    for i,y in enumerate(train_preds):
        print(i, y)
        if i>20: break

    training_score, numerator, denominator = eval_wrapper_with_matrices(train_preds, df[1]['Response'])
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
    offset_train_preds = np.vstack((train_preds, train_preds, df[1]['Response'].values))
    for j in range(num_classes):
        train_offset = lambda x: -apply_offset(offset_train_preds, x, j)
        offsets[j] = fmin_powell(train_offset, offsets[j])  

    print("after training offsets: %.3f seconds" % (time.time() - start_time))
    
    # apply offsets to test
    data = np.vstack((test_preds, test_preds, df[3]['Response'].values))
    for j in range(num_classes):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

    final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

    preds_out = pd.DataFrame({"Id": df[3]['Id'].values, "Response": final_test_preds})
    preds_out = preds_out.set_index('Id')
    preds_out.to_csv(filenames[3]['out'])
    print("file created: ", filenames[3]['out'])

    # apply offsets to training set
    data = np.vstack((train_preds, train_preds, df[1]['Response'].values))
    for j in range(num_classes):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

    final_train_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

    train_preds_out = pd.DataFrame({"Id": df[1]['Id'].values, "Response": final_train_preds})
    train_preds_out = train_preds_out.set_index('Id')
    train_preds_out.to_csv(filenames[1]['out'])
    print("file created: ", filenames[1]['out'])

if __name__ == "__main__":
    main()
    print("end: %.3f seconds" % (time.time() - start_time))
