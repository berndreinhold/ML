import time
import pandas as pd 
import numpy as np 
import random

#from https://www.kaggle.com/chenglongchen/prudential-life-insurance-assessment/prudential-seed-23/code
#This script has been released under the Apache 2.0 open source license. 

#this is the modified version, different from the website, which I adjusted to understand what is going on.

random.seed(123)
start_time = time.time()

array_variables = ['Employment_Info', 'Family_Hist', 'Insurance_History', 'InsuredInfo', 'Medical_History', 'Medical_Keyword', 'Product_Info'] #list the variables with _[0-9]+ in order to count the number of nan per 'array' and to aggregate them

minmax = [(1,2),(3,4),(5,6),(7,8)]

def main():
    print("Load the data using pandas")
    #all_data = pd.read_csv("/home/reinhold/data/ML/Prudential/input_data/Prudential_1000.csv", header=0)
    train = pd.read_csv("/home/reinhold/data/ML/Prudential/input_data/train_Prudential.csv", header=0)
    predicted_labels = pd.read_csv("/home/reinhold/data/ML/Prudential/output_data/py_xgb_train_afterPCA_Feb14.csv", header=0)

    output_name = "/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_pred_resp%d-%d.csv" % (minmax[0][0], minmax[0][1])
    output_labels_name = "/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_predicted_labels_pred_resp%d-%d.csv" % (minmax[0][0], minmax[0][1])

    print(train.shape)
    print(predicted_labels.shape)
    #include the neigboring responses:
    unique_responses = train['Response'].unique()
    if min(unique_responses)<1: print("unexpected min response value: %d" % min(unique_responses))
    if max(unique_responses)>8: print("unexpected max response value: %d" % max(unique_responses))

    minmax_ = (max(minmax[0][0]-1, min(unique_responses)), min(minmax[0][1]+1, max(unique_responses)))
    #minmax_ = (minmax[0]-1, minmax[1]+1)
    #print(type(minmax[0]))
    print(minmax_)
    select_subset(minmax_, train, predicted_labels, output_name, output_label_name)


#def select_subset(minmax_, train, actual_labels, predicted_labels):

def select_subset(minmax_, train, predicted_labels, output_train_name, output_train_labels_name):
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
    out_train = train.join(predicted_labels, how='inner', rsuffix='_pred')
    print(out_train.shape)

    #for col in out_train.columns:
    #    print(col)

    #print(out_train)

    out_train = out_train[(out_train['Response']>=minmax_[0]) & (out_train['Response_pred']>=minmax_[0]) & (out_train['Response']<=minmax_[1]) & (out_train['Response_pred']<=minmax_[1])].copy()
    #out_train = out_train[(out_train['Response']>=minmax_[0]) & (out_train['Response']<=minmax_[1])].copy()
    #out_train = out_train[(out_train['Response']>=1) & (out_train['Response']<=3) & (out_train['Response_pred']>=1) & (out_train['Response_pred']<=3)].copy()

    out_train.set_index('Id')

    #store labels in separate output file:
    out_train_labels = pd.DataFrame({"Id": out_train['Id_pred'].astype(int).values, "Response": df_train['Response_pred'].astype(int).values})
    out_train_labels.set_index('Id')
    out_train_labels.to_csv(output_train_labels_name, index=False) #Id is the index

    out_train.drop(['Id_pred','Response_pred'], axis=1)
    #write output:
    out_train.to_csv(output_train_name, index=False) #Id is the index


    #print(train.filter(regex="count_null").head(10))
    
    print("output file created: %s" % output_train_name)
    print(out_train.shape)



if __name__ == "__main__":
    #drop_array_variables()
    #clean_data()
    main()
    print("end: %.3f seconds" % (time.time() - start_time))
