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

def count_null(df):

    #for col in df.columns:
    #    found = 0
    #    for prefix in array_variables:
    #        if col.startswith(prefix): found = 1 
    #    print(col, found)


    #print(df.info())
    #print (df.filter(regex=array_variables[0]).head(3))
    #print(df)
    #print(df.describe())
    #return

    #h = [col for col in df.head(10).columns if df[col].isnull()]
    #print(h)

    #h = df.isnull().astype(int).head(10).sum(0)
    #print(sorted(h[h>0].index)) #print the column name, if the number of isnull() is > 0
    #print(h)

    #print(df.isnull().astype(int).sum(1).head(10))
    #print(df[sorted(h[h>0].index)].head(10))
    #verified that these queries do what I think they do.

    df['all_count_null'] = df.isnull().astype(int).sum(1)

    for av in array_variables:
        #http://stackoverflow.com/questions/12569730/sum-all-columns-with-a-wildcard-name-search-using-python-pandas#12570410
        df['%s_count_null' % av] = df.filter(regex=av).isnull().astype(int).sum(1)

    #for col in df.columns:
    #    print(col)

    #print(df.filter(regex="count_null").head(10))

def remove_vars(df):
    in_ = []
    out_ = [] #list of columns, that are excluded from the usual correlation matrix: all array_variables, except their aggregates (var_count_null, var_sum)
    for col in df.columns:
        if col.find("count_null")!=-1 or col.find("sum")!=-1:
            in_[len(in_):] = [col]
        for av in array_variables:
            if col.startswith(av):
                out_[len(out_):] = [col]

    all_set = set(df.columns)
    in_set = set(in_)
    out_set = set(out_)

    all_columns = all_set - (out_set - in_set) #double exclusion, yeah! first out_set - in_set: exclude count_null, sum from array_variables, because we want to keep these in the overall dataset.

    column_list = [x for x in all_columns]
    column_list.sort()
    print(column_list)

    
    return df[column_list]


    
def drop_array_variables():
    """
    try without the array variables, just the sum and the not null counts - this is because Brendan had concerns about too many variables with a boosted decision tree.
    Has to be called after clean_data(), since this creates clean output dataset.
    """
    train = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_cleaned.csv", header=0)
    test = pd.read_csv("/home/reinhold/data/ML/Prudential/intermediate_data/test_Prudential_cleaned.csv", header=0)

    train = remove_vars(train)
    test = remove_vars(test)

    #store output

    train_buffer = "/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_stripped_array_variables.csv"
    test_buffer = "/home/reinhold/data/ML/Prudential/intermediate_data/test_Prudential_stripped_array_variables.csv"

    train.set_index('Id')
    test.set_index('Id')
    train.to_csv(train_buffer, index=False) #Id is the index
    test.to_csv(test_buffer, index=False) #Id is the index



def clean_data():
    """
    fill NaN with median and scale by max-value
    """
    
    print("Load the data using pandas")
    #all_data = pd.read_csv("/home/reinhold/data/ML/Prudential/input_data/Prudential_1000.csv", header=0)
    train = pd.read_csv("/home/reinhold/data/ML/Prudential/input_data/train_Prudential.csv", header=0)
    test = pd.read_csv("/home/reinhold/data/ML/Prudential/input_data/test_Prudential.csv", header=0)

    print(train.shape)
    print(test.shape)

    # combine train and test
    all_data = train.append(test)

    #count_null(all_data) #adds new variables

    # factorize categorical variables -> "D2" to unique identifier
    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Ins_AgeBMI'] = all_data['Ins_Age']*all_data['BMI']


    ####fill missing values with medians and scale each column by its max_value
    medians = all_data.median(0)
    max_values = all_data.max(0)
    #print(medians)
    #print(max_values)

    for i in medians.index:
        if i == "Response":
            print("skip 'Response'")
            continue
        all_data[i].fillna(medians[i], inplace=True)

    print('Eliminate missing values')    
    # Use -1 for any others
    all_data.fillna(-1, inplace=True)

    for mv in max_values.index:
        if max_values[mv]>0 and not (mv ==  'Response' or mv == 'Id'):
            all_data[mv] = all_data[mv]/max_values[mv];
        else: print("unexpected max value %f for index %s, do nothing" % (max_values[mv], mv))

    ##now aggregate all array_variables, now that they are all normalized to a max value of 1
    #for av in array_variables:
    #    #http://stackoverflow.com/questions/12569730/sum-all-columns-with-a-wildcard-name-search-using-python-pandas#12570410
    #    all_data['%s_sum' % av] = all_data.filter(regex=av).sum(1)
    #comment: PCA showed that these are strongly (anti)correlated with their individual array variables and so I disabled them again - it was interesting to study them though
        

    print('nan-values filled and sums for array variables calculated: %.3f seconds' % (time.time() - start_time))    

    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)
    
    # split train and test
    train = all_data[all_data['Response']>0].copy()
    test = all_data[all_data['Response']<1].copy()

    #standardizing datasets
    #calculate means and stddev from train dataset and apply to both training and test dataset, except on 'Id' and 'Response'

    columns_ = [x for x in train.columns if train.std()[x]==0]
    print("drop columns with std==0:")
    print(columns_)
    train = train.drop(columns_, axis=1)
    test = test.drop(columns_, axis=1)

    columns_set = set([x for x in train.columns if train.std()[x]>0]) 
    columns_ = [x for x in columns_set - set(['Id', 'Response'])] #set is useful to exclude certain columns
    print(columns_)


    #for col in train.columns:
    #    if col in ['Id', 'Response']:
    #        print("skip %s" % col)
    #        continue
    #    if train.std()[col]==0:
    #        print("skip %s, since std is 0" % col)
    #        continue
    #    train[col] = (train[col] - train.mean()[col]) / train.std()[col]
    #    test[col] = (test[col] - train.mean()[col]) / train.std()[col]

    means_ = train.mean()[columns_]
    std_ = train.std()[columns_]

    train[columns_] = (train[columns_] - means_[columns_]) / std_[columns_]
    test[columns_] = (test[columns_] - means_[columns_]) / std_[columns_]


    #train_norm['Id'] = train.std()['Id']*train_norm['Id'] + train.mean()['Id']
    #train_norm['Response'] = train.std()['Response']*train_norm['Response'] + train.mean()['Response']
        


    #store output
    train_buffer = "/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_standardized.csv"
    test_buffer = "/home/reinhold/data/ML/Prudential/intermediate_data/test_Prudential_standardized.csv"
    means_buffer = "/home/reinhold/data/ML/Prudential/intermediate_data/means_Prudential_standardized.csv"

    train.set_index('Id')
    test.set_index('Id')
    train.to_csv(train_buffer, index=False) #Id is the index
    test.to_csv(test_buffer, index=False) #Id is the index

    means_df = pd.DataFrame({'mean':means_, 'std':std_})
    means_df.to_csv(means_buffer, index=False)

    for col in train.columns:
        print(col)

    #print(train.filter(regex="count_null").head(10))
    
    print("output file created: %s, entries: %d" % (train_buffer, len(train)))
    print("output file created: %s, entries: %d" % (test_buffer, len(test)))
    print("output file created: %s, entries: %d" % (means_buffer, len(means_df)))


if __name__ == "__main__":
    #drop_array_variables()
    clean_data()
    print("end: %.3f seconds" % (time.time() - start_time))
