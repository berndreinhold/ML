# ML
machine learning using python

Jan 29th, 2016:
look into Machine learning tools/software, see if these algorithms can be helpful to improve the electron recoil/ nuclear recoil discrimination.
Tools are scikit_learn, python pandas, etc.

Tutorials related to kaggle competitions for the Prudential competition.

pipeline:
---------
#1. clean up of dataset: add missing values, transform strings to integer values, etc. 
1. all variables in the training dataset are already normalized: train_Prudential_reduced.csv (after extracting a cross validation dataset: 40000 lines) 
-> this might be a good step to introduce new variables
-> median to fill missing entries
1a. get cross validation set from training set: (19382 lines) cv_Prudential.csv
[1b. construct additional features features]
2. visualize variables before anything
3. PCA
4. k-means to find clusters in data (or k-nearest neighbors)
5. PCA per cluster
6. random forest
6a. linearSVC