""" copy of http://www.scipy-lectures.org/advanced/scikit-learn/, iris_alldim.py
adjusted for Prudential dataset
Author : Bernd
Date : Feb. 2, 2016
"""
import numpy as np
import pandas as pd 
from sklearn import datasets
#import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import svm
from sklearn import decomposition
import seaborn as sns
import math

minmax = [(1,2),(3,4),(5,6),(7,8)]

def main():

    path = '/home/reinhold/data/ML/Prudential/intermediate_data/'
    #train = { 'in': path + 'train_Prudential_standardized.csv',
    #          'out': path + 'train_Prudential_afterPCA.csv'}
    #test = { 'in': path + 'test_Prudential_standardized.csv',
    #         'out': path + 'test_Prudential_afterPCA.csv'}

    for i in minmax:
        train = { 'in': path + 'train_Prudential_pred_resp%d-%d.csv' % i,
                  'out': path + 'train_Prudential_pred_afterPCA_resp%d-%d.csv' % i}
        test = { 'in': path + 'test_Prudential_pred_resp%d-%d.csv' % i,
                 'out': path + 'test_Prudential_pred_afterPCA_resp%d-%d.csv' % i}
        loop(train, test)




def loop(train, test):

    #training dataset:
    #df_train = pd.read_csv('/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_cleaned.csv', header=0)
    df_train = pd.read_csv(train['in'], header=0)
    print("unique values df_train:")
    print(df_train['Response'].unique())
    
    #df_train_buffer = pd.DataFrame(df_train[['Id','Response']], columns=['Id', 'Response'], index=df_train['Id']) #copy them before they are being dropped, for later insertion in the output dataframe
    df_train_buffer = pd.DataFrame({"Id": df_train['Id'].astype(int).values, "Response": df_train['Response'].astype(int).values})
    #df_train_buffer = df_train_buffer.set_index('Id')
    print(df_train_buffer.describe())
    print("unique values df_train_buffer:")
    print(df_train_buffer['Response'].unique())
    print(df_train_buffer['Id'].unique())

    df_train = df_train.drop(['Response', 'Id'], axis=1)

    #test dataset:
    #df_test = pd.read_csv('/home/reinhold/data/ML/Prudential/intermediate_data/test_Prudential_cleaned.csv', header=0)
    df_test = pd.read_csv(test['in'], header=0)
    #df_test_buffer = pd.DataFrame(df_test[['Id','Response']], columns=['Id', 'Response'], index=df_test['Id']) #copy them before they are being dropped, for later insertion in the output dataframe
    df_test_buffer = pd.DataFrame({"Id": df_test['Id'].astype(int).values, "Response": df_test['Response'].astype(int).values})
    print("unique values df_test_buffer:")
    print(df_test_buffer['Response'].unique())
    print(df_test_buffer['Id'].unique())

    df_test = df_test.drop(['Response', 'Id'], axis=1)

    #fit

    #print("selected number of components: out of %d" % len(df_train.columns))
    #for i in range(50,99):
    #pca = decomposition.PCA(n_components=i*1./100)
    print("do PCA fit to training set:")
    pca = decomposition.PCA()
    pca.fit(df_train) #fit the model with X - what does that mean?
    #print("%d, %d" % (i, pca.n_components_)) 
    #print(pca.explained_variance_)
    #print(pca.explained_variance_ratio_)

    #buffer = pca.transform(X) #apply the dimensionality reduction on X
    #eigen_val, eigen_vec = np.linalg.eig(np.cov(buffer.transpose()))
    #print "eigen_val, eigen_vec (2)"
    #print eigen_val
    #print eigen_vec
    

    #transform
    df_train_PCA = pd.DataFrame(pca.transform(df_train))
    df_train_PCA['Id']= df_train_buffer['Id'] 
    df_train_PCA['Response']= df_train_buffer['Response']
    df_train_PCA.set_index('Id')
    df_train_PCA.to_csv(train['out'], index=False)
    
    df_test_PCA = pd.DataFrame(pca.transform(df_test))
    df_test_PCA['Id']= df_test_buffer['Id'] 
    df_test_PCA['Response']= df_test_buffer['Response'] 
    df_test_PCA.set_index('Id')
    df_test_PCA.to_csv(test['out'], index=False)

    print("output files: %s and %s"% (train['out'], test['out']))

    #print(type(eigen_val_matrix))
    #print(eigen_val_matrix.size)

    #df_matrix = pd.DataFrame(eigen_val_matrix)

    #print(df_matrix.describe())

    fig = plt.figure()
    #df_matrix.hist()
    #df_matrix.hist()
    ax = sns.heatmap(pca.get_covariance(), vmax=1, square=True, cbar=True)
    fig.autofmt_xdate(rotation=70) #does not exist for y-axis!
    #ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0) #http://stackoverflow.com/questions/10998621/rotate-axis-text-in-python-matplotlib
    plt.savefig("corr_matrix_afterPCA.png")
    print("output: corr_matrix_afterPCA.png")
        
if __name__ == "__main__":
    main()
