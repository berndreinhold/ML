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

def main():

    #training dataset:
    #df_train = pd.read_csv('/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_cleaned.csv', header=0)
    df_train = pd.read_csv('/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_standardized.csv', header=0)
    df_results = pd.DataFrame()
    outname_results = '/home/reinhold/data/ML/Prudential/intermediate_data/train_PCA_variance_ratio.csv'

    for r in range(1,9):
        outname_train = '/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_std_r%d.csv' % r
        PCA_per_response(df_train[df_train['Response']==r], r, df_results, outname_train)

    df_results.to_csv(outname_results)
    print(outname_results)

def PCA_per_response(df_train, response, df_results, outname):
    df_train_buffer = pd.DataFrame(df_train[['Id','Response']], columns=['Id', 'Response'], index=df_train['Id']) #copy them before they are being dropped, for later insertion in the output dataframe
    df_train = df_train.drop(['Response', 'Id'], axis=1)

    pca = decomposition.PCA() #keep all variables
    pca.fit(df_train)

    #print(pca.explained_variance_)
    df_results['response_%d'% response]=pca.explained_variance_ratio_

    df_train_PCA = pd.DataFrame(pca.transform(df_train))
    df_train_PCA['Id']= df_train_buffer['Id'] 
    df_train_PCA['Response']= df_train_buffer['Response']
    df_train_PCA.set_index('Id')
    df_train_PCA.to_csv(outname, index=False)
    print(response, len(df_train), outname)

    #print(pca.components_.T)
    fig = plt.figure()
    ax = sns.heatmap(pca.components_.T, vmax=1, xticklabels=5, yticklabels=5, square=True, cbar=True)
    ax.set_title("PCA Transformation matrix (response %d)" % response)
    ax.set_xlabel("vars after PCA transformation")
    ax.set_ylabel("vars before PCA transformation")
    fig.autofmt_xdate(rotation=70) #does not exist for y-axis!
    ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0)
    plt.savefig("PCA_T_response%d.png" % response)

if __name__ == "__main__":
    main()
