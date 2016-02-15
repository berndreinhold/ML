import time
import pandas as pd 
import numpy as np 
import random

#from https://www.kaggle.com/chenglongchen/prudential-life-insurance-assessment/prudential-seed-23/code
#This script has been released under the Apache 2.0 open source license. 

#this is the modified version, different from the website, which I adjusted to understand what is going on.

start_time = time.time()

minmax = [(1,2),(3,4),(5,6),(7,8)]

def main():

    for i in minmax:
        loop(i)


def loop(minmax_):


    path =  ('/home/reinhold/data/ML/Prudential/intermediate_data/',
             '/home/reinhold/data/ML/Prudential/output_data/')

    train = { 'in': path[0] + 'train_Prudential_standardized.csv',
              'out': path[0] + 'train_Prudential_pred_resp%d-%d.csv' % (minmax_[0], minmax_[1])}
    test = { 'in': path[0] + 'test_Prudential_standardized.csv',
             'out': path[0] + 'test_Prudential_pred_resp%d-%d.csv' % (minmax_[0], minmax_[1])}

    train_labels = { 'in': path[1] + 'py_xgb_train_afterPCA_Feb14.csv',
                     'out': path[0] + "train_Prudential_predicted_labels_pred_resp%d-%d.csv" % (minmax_[0], minmax_[1])}

    test_labels = { 'in': path[1] + 'py_xgb_afterPCA_Feb14.csv',
                     'out': path[0] + "test_Prudential_predicted_labels_pred_resp%d-%d.csv" % (minmax_[0], minmax_[1])}

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

    df_train = df[0]
    #include the neigboring responses:
    unique_responses = df_train['Response'].unique()
    if min(unique_responses)<1: print("unexpected min response value: %d" % min(unique_responses))
    if max(unique_responses)>8: print("unexpected max response value: %d" % max(unique_responses))

    minmax_ = (max(minmax_[0]-1, min(unique_responses)), min(minmax_[1]+1, max(unique_responses)))
    #minmax_ = (-1, minmax[1]+1)
    #print(type(minmax[0]))
    print(minmax_)
    #select_subset(minmax_, df_train, df_predicted_labels, output_name, output_labels_name)
    select_subset(minmax_, df, filenames)


#def select_subset(minmax_, train, actual_labels, predicted_labels):

def select_subset(minmax_, df, filenames):
    """
    select subset of the whole training set in order to perform
    parameters:
    -----------
    minmax_: is a tuple defining the selection range, e.g. (1,2)
    train: the training dataframe
    actual_labels: the dataframe of actual labels (to be trained on)
    predicted_labels: labels as predicted by a previous round of boosted decision tree
    """
    #join the train and the predicted_labels df
    out_train = df[0].join(df[1], how='inner', rsuffix='_pred')
    print(out_train.shape)

    out_test = df[2].join(df[3], how='inner', rsuffix='_pred')
    print(out_test.shape)

    #for col in out_train.columns:
    #    print(col)

    #print(out_train)

    out_train = out_train[(out_train['Response']>=minmax_[0]) & (out_train['Response_pred']>=minmax_[0]) & (out_train['Response']<=minmax_[1]) & (out_train['Response_pred']<=minmax_[1])].copy()
    out_train.set_index('Id')
    df[0] = out_train

    #store labels in separate output file:
    out_train_labels = pd.DataFrame({"Id": out_train['Id_pred'].astype(int).values, "Response": out_train['Response_pred'].astype(int).values})
    out_train_labels.set_index('Id')
    out_train_labels.to_csv(filenames[1]['out'], index=False) #Id is the index
    df[1] = out_train_labels

    #for the test dataset the Response field contains -1
    out_test = out_test[(out_test['Response_pred']>=minmax_[0]) & (out_test['Response_pred']<=minmax_[1])].copy()
    out_test.set_index('Id')
    df[2] = out_test
        
    out_train.drop(['Id_pred','Response_pred'], axis=1)
    #write output:
    out_train.to_csv(filenames[0]['out'], index=False) #Id is the index

    #store labels in separate output file:
    out_test_labels = pd.DataFrame({"Id": out_test['Id_pred'].astype(int).values, "Response": out_test['Response_pred'].astype(int).values})
    out_test_labels.set_index('Id')
    out_test_labels.to_csv(filenames[3]['out'], index=False) #Id is the index
    df[3] = out_test_labels

    out_test.drop(['Id_pred','Response_pred'], axis=1)
    #write output:
    out_test.to_csv(filenames[2]['out'], index=False) #Id is the index


    #print(train.filter(regex="count_null").head(10))

    for i,fn in enumerate(filenames):
        print("output file created: %s" % fn['out'])
        print(df[i].shape)



if __name__ == "__main__":
    #drop_array_variables()
    #clean_data()
    main()
    print("end: %.3f seconds" % (time.time() - start_time))
