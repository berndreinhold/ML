""" copy of http://www.scipy-lectures.org/advanced/scikit-learn/, iris_alldim.py
adjusted for Prudential dataset
Author : Bernd
Date : Feb. 2, 2016
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import decomposition
import seaborn as sns
import math
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

#following implementation in http://pastebin.com/JuEmXqAv

def main():
    for resp in [1,2, 3]:
        for n_comp in [3, 5, 10]:
            kMeansScan(resp, n_comp, "/home/reinhold/data/ML/Prudential/intermediate_data/figures/")
    
def kMeansScan(resp, n_comp, fig_dir=""):
    
    #training dataset:
    df_train = pd.read_csv('/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_standardized.csv', header=0)
    df_train = df_train[df_train['Response']==resp] #only one Response

    df_train_buffer = pd.DataFrame(df_train[['Id','Response']], columns=['Id', 'Response'], index=df_train['Id']) #copy them before they are being dropped, for later insertion in the output dataframe
    df_train = df_train.drop(['Response', 'Id'], axis=1)
    #print(df_train.columns)

    #PCA
    pca = decomposition.PCA(n_components=n_comp)
    pca.fit(df_train) #fit the model with X - what does that mean?
    #print(pca.explained_variance_)
    #print(pca.explained_variance_ratio_)

    #transform
    df_train_PCA = pd.DataFrame(pca.transform(df_train))

    #print("now kMean:", df_train_PCA.shape)

    from scipy.spatial.distance import cdist, pdist
    clusters = [2,3,4,6,8]
    meandist, meandist2 = [],[]
    between_cluster_sumsquares = [] #http://www.slideshare.net/SarahGuido/kmeans-clustering-with-scikitlearn, p.20
    silhouette_s = []


    for k in clusters:
        model = KMeans(n_clusters=k)
        model.fit(df_train_PCA)
        clusassign = model.predict(df_train_PCA)
        #print(cdist(df_train_PCA, model.cluster_centers_, 'euclidean'))
        #print(model.cluster_centers_)
        dist_within_cluster = np.min(cdist(df_train_PCA, model.cluster_centers_, 'euclidean'), axis=1)
        #print(k)
        #print(len(model.labels_))
        #print(model.labels_)
        meandist.append(sum(dist_within_cluster)/df_train_PCA.shape[0])
        #meandist2.append(sum(dist_within_cluster**2))
        #between_cluster_sumsquares.append(sum(pdist(df_train_PCA)**2)/df_train_PCA.shape[0]-meandist2[-1])
        #print(df_train_PCA.shape[0])
        ss = metrics.silhouette_score(df_train_PCA, model.labels_)
        silhouette_s.append(ss) #Segmentation fault (core dumped)
        if ss >= max(silhouette_s): best_model=model
        
        #print(line)
        #print("OK")

    print(meandist)
    #print(between_cluster_sumsquares)
    print(resp, df_train_PCA.shape, ", ".join(["%.5f" % ss for ss in silhouette_s]))


    """
    Plot average distance from observations from the cluster centroid
    to use the Elbow Method to identify number of clusters to choose
    """
    plt.plot(clusters, meandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')
    #plt.show()
    plt.savefig("%s/resp%d_ncomp%d_avg_distance.png" %(fig_dir,resp,n_comp))

    #between_cluster_sumsquares was a very big number 10**4 -> something is not quite right
    #plt.plot(clusters, between_cluster_sumsquares)
    #plt.xlabel('Number of clusters')
    #plt.ylabel('between clusters sum squares')
    #plt.title('')
    #plt.show()

    plt.plot(clusters, silhouette_s)
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette_score')
    plt.title('Silhouette score')
    #plt.show()
    plt.savefig("%s/resp%d_ncomp%d_silhouette_score.png" %(fig_dir,resp,n_comp))
    
    # plot clusters
    nplots=4
    if n_comp==3: nplots=3
    pca = decomposition.PCA(nplots)
    plot_columns = pca.fit_transform(df_train_PCA)
    if nplots==4:
        plt.subplot(221)
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=best_model.labels_,)
        plt.xlabel('Canonical variable 0')
        plt.ylabel('Canonical variable 1')

        plt.subplot(222)
        plt.scatter(x=plot_columns[:,1], y=plot_columns[:,2], c=best_model.labels_,)
        plt.xlabel('Canonical variable 1')
        plt.ylabel('Canonical variable 2')
        
        plt.subplot(223)
        plt.scatter(x=plot_columns[:,2], y=plot_columns[:,3], c=best_model.labels_,)
        plt.xlabel('Canonical variable 2')
        plt.ylabel('Canonical variable 3')
        
        plt.subplot(224)
        plt.scatter(x=plot_columns[:,3], y=plot_columns[:,0], c=best_model.labels_,)
        plt.xlabel('Canonical variable 3')
        plt.ylabel('Canonical variable 0')

    else:
        plt.subplot(221)
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=best_model.labels_,)
        plt.xlabel('Canonical variable 0')
        plt.ylabel('Canonical variable 1')

        plt.subplot(222)
        plt.scatter(x=plot_columns[:,1], y=plot_columns[:,2], c=best_model.labels_,)
        plt.xlabel('Canonical variable 1')
        plt.ylabel('Canonical variable 2')
        
        plt.subplot(223)
        plt.scatter(x=plot_columns[:,2], y=plot_columns[:,0], c=best_model.labels_,)
        plt.xlabel('Canonical variable 2')
        plt.ylabel('Canonical variable 0')
    
    plt.suptitle('Scatterplot of most relevant variables (best model: %d clusters, after PCA, silhouette score: %.4f)' % (len(best_model.cluster_centers_), max(silhouette_s)))
    #plt.show()
    plt.savefig("%s/resp%d_ncomp%d_%dmostcritvars.png" %(fig_dir,resp,n_comp,nplots))

        
if __name__ == "__main__":
    main()
