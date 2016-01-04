""" copy of http://www.scipy-lectures.org/advanced/scikit-learn/
Author : Bernd
Date : Dec. 23, 2015
"""
import numpy as np
from sklearn import datasets
#import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import svm
from sklearn import decomposition
import math

def digits_dataset():
    #digits
    digits = datasets.load_digits()
    print "digits.images.shape[0]:", digits.images.shape[0]
    #pl.imshow(digits.images[0], cmap=pl.cm.gray_r) 
    #pl.savefig("digit.png")
    data = digits.images.reshape((digits.images.shape[0], -1))
    print "after reshape:", data
    
def PCA(X, n_compnts=None):
    print "PCA with %s components" % n_compnts
    #eigen_val, eigen_vec = np.linalg.eig(np.cov(X.transpose()))
    #print "eigen_val, eigen_vec (1)"
    #print eigen_val
    #print eigen_vec
    pca = decomposition.PCA(n_components=n_compnts)
    pca.fit(X) #fit the model with X - what does that mean?
    print "pca.components_.T:"
    print pca.components_.T
    print "pca.mean_:", pca.mean_ 
    return pca.transform(X), pca.components_.T
    #pca.components_
    #print pca.explained_variance_ratio_
    #buffer = pca.transform(X) #apply the dimensionality reduction on X
    #eigen_val, eigen_vec = np.linalg.eig(np.cov(buffer.transpose()))
    #print "eigen_val, eigen_vec (2)"
    #print eigen_val
    #print eigen_vec
    #return buffer

def main():
    iris = datasets.load_iris()
    #print iris
    #print type(iris)

    #print "iris.data:"
    #eigen_val, eigen_vec = np.linalg.eig(np.cov(iris.data.transpose()))
    #print iris.data
    #print [sum(iris.data[:,k]*eigen_vec[k,:] for k in range(len(eigen_vec)))]

    X, eigen_vec_matrix=PCA(iris.data, 4) #ASSUMP n_components
    #X=PCA(iris.data, 0.5) #ASSUMP n_components
    #print "X:"
    #print X

    #pl.imshow(data[0], cmap=pl.cm.gray_r) 
    #pl.savefig("digit2.png")
    #iris again
    #return

    print "linearSVC"
    clf = svm.LinearSVC(C=1) #an estimator, C is the regularization parameter #ASSUMP C=1
    clf.fit(X, iris.target) # learn from the data 

    #print "prediction: ", clf.predict([[ 5.0,  3.6,  1.3,  0.25]])
    #print "coefficients: ", clf.coef_

    step = .02  # step size in the mesh #ASSUMP step
    var_pairs = [(0,1), (1,2), (2,3), (3,0)] #(variable on x-axis, variable on y-axis)
    label=["sepal length", "sepal width", "petal length", "petal width"]

    label2= [None] * len(label)
    for k in range(len(label2)):
        label2[k]="%.2f*%s_c+%.2f*%s_c+%.2f*%s_c+%.2f*%s_c" % (eigen_vec_matrix[0,k],label[0], eigen_vec_matrix[1,k],label[1], eigen_vec_matrix[2,k],label[2], eigen_vec_matrix[3,k],label[3])
        label2[k]=label2[k].replace('+-','-')

    label=['var 0', 'var 1', 'var 2', 'var 3']

    var_text = [None] * len(label)
    for k in range(len(label)):
        var_text[k]= "%s: %s" % (label[k], label2[k])

    var_text = "\n".join(var_text)

    # determine min and max for all four variables
    var_min, var_max = [X[:, k].min() - 0.5 for k in range(len(var_pairs))], [X[:, k].max() + 0.5 for k in range(len(var_pairs))]

    #plot_var = "mean" #ASSUMP
    plot_var = "stddev" #ASSUMP

    #plot configuration
    plt.suptitle("%s(predicted value), iris subcategories vs. observables (marginalized over vars not shown)" % plot_var)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    G = gridspec.GridSpec(3,2)

    #produce all 4 plots for four pairs of variables
    for i, j in enumerate(var_pairs):

        if i==0:
            plt.subplot(G[2,:])
            plt.xticks(())
            plt.yticks(())
            plt.text(0.02,0.5, var_text, ha='left', va='center',size=10)
            plt.subplot(G[0,0])
        elif i==1: plt.subplot(G[0,1])
        elif i==2: plt.subplot(G[1,0])
        elif i==3: plt.subplot(G[1,1])
 

        #xx and yy are for the contours of two specific variables
        xx, yy = np.meshgrid(np.arange(var_min[j[0]], var_max[j[0]], step), np.arange(var_min[j[1]], var_max[j[1]], step))

        #x are all 4 variables
        x = np.ones((4, len(xx.ravel())))
        x[j[0]]=xx.ravel()
        x[j[1]]=yy.ravel()
        #x1=xx.ravel()
        #x2=yy.ravel()
        #x3=xx.ravel()
        #x4=yy.ravel()

        zz_buffer = np.zeros(len(xx.ravel()))
        zz_average = np.zeros(len(xx.ravel()))
        zz_stddev = np.zeros(len(xx.ravel()))
        zz_sum_squared = np.zeros(len(xx.ravel()))
        n=100 #1000 takes significantly longer than 100, not just a factor of 10! #ASSUMP n
        for k in range(n): #n random numbers for the variable to marginalize over
            for m in range(4):
                if m!=j[0] and m!=j[1]:
                    x[m]=((var_max[m]-var_min[m])*np.random.random()+var_min[m]) * np.ones(len(xx.ravel()))
                    #x[m]=((var_max[m]-var_min[m])*0.5+var_min[m])
                #print type(x[m])
            zz_buffer = clf.predict(np.c_[x[0],x[1],x[2],x[3]])
            zz_average = zz_average + zz_buffer
            zz_sum_squared = zz_sum_squared + [zz_buffer[l]*zz_buffer[l] for l in range(len(zz_buffer))]
        zz_average = zz_average/n
        zz_stddev = zz_sum_squared/n - [zz_average[l]*zz_average[l] for l in range(len(zz_average))]
        zz_stddev = np.zeros(len(xx.ravel())) + [math.sqrt(zz_stddev[l]) for l in range(len(zz_stddev))] #so that zz_stddev remains a ndarray
        zz_average = zz_average.reshape(xx.shape)
        zz_stddev = zz_stddev.reshape(xx.shape)
        #cf = plt.contourf(xx, yy, zz_average, cmap=plt.cm.Paired, alpha=0.8)
        if plot_var =="mean": cf = plt.contourf(xx, yy, zz_average, cmap=plt.cm.hot, alpha=0.8)
        else: cf = plt.contourf(xx, yy, zz_stddev, cmap=plt.cm.hot, alpha=0.8)
        #C = plt.contour(xx, yy, zz_average, colors='black', linewidth=.5)
        #plt.clabel(C, inline=1, fontsize=10)
        plt.colorbar(cf)
        
        plt.scatter(X[:, j[0]], X[:, j[1]], c=iris.target, cmap=plt.cm.hot)
        plt.xlabel(label[j[0]])
        plt.ylabel(label[j[1]])


    plt.show()

if __name__ == "__main__":
    main()
