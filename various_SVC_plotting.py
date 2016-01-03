"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""
print __doc__

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .5  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    #plt.title("baeh")
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    #plt.subplots_adjust(wspace=.1, hspace=.1)

    print "xx%%%%%%%%%%%%%%%"
    print xx #2D array
    print "X[:,0]%%%%%%%%%%%%%%%"
    print X[:,0] #2D array
    print "xx.ravel()%%%%%%%%%%%%%%%"
    print xx.ravel() #1D array
    print "%%%%%%%%%%%%%%%"
    print type(np.c_[xx.ravel(), yy.ravel()])
    print "np.c_1%%%%%%%%%%%%%%%"
    print np.c_[xx.ravel(), yy.ravel()]
    print "np.c_2%%%%%%%%%%%%%%%"
    print np.c_[xx.ravel()]
    #help(clf.predict)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z = clf.predict(np.c_[xx.ravel()])
    print "len 1: ", len(np.c_[xx.ravel(), yy.ravel()])
    print "len 2: ", len(np.c_[xx.ravel()])
    print "len Z: ", len(Z)

    # Put the result into a color plot
    print "Z before reshape%%%%%%%%%%%%%%%"
    print Z
    Z = Z.reshape(xx.shape)
    print "Z after reshape%%%%%%%%%%%%%%%"
    print Z
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks()
    plt.yticks()
    plt.title(titles[i])

plt.show()
