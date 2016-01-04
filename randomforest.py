"""
get the training set into shape as well as the test set, and then apply the random forest, which is the easy part.
adjusted from https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests and myfirstforest.py
Author : Bernd
Date : Dec. 30, 2015
"""

print(__doc__)
import pandas as pd
import numpy as np
import pylab as p
import sys
import csv
from sklearn.ensemble import RandomForestClassifier

def fillMissingValues_NewVars(df, median_ages=None):

    string2Enumeration(df) #required for calculation of median_ages and applying age-correction
    # All missing Embarked -> just make them embark from most common place
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
    
    #fill in ages based on Pclass and Sex information
    if median_ages==None:
        median_ages = np.zeros([2,3])

        for i in range(0,2):
            for j in range(0,3):
                median_ages[i,j]=df[(df['Sex_int']==i) & (df['Pclass']==j+1)]['Age'].median()

        print "median age per Sex and Pclass:"
        print median_ages

    df['AgeFill'] = df['Age']

    for i in range(0,2):
        for j in range(0,3):
            df.loc[(df.Age.isnull()) & (df.Sex_int==i) & (df.Pclass==j+1), 'AgeFill'] = median_ages[i,j]

    df['AgeIsNull']=df['Age'].isnull().astype(int)
    #print df[df['Age'].isnull()][['Sex_int','Pclass', 'Age', 'AgeFill', 'AgeIsNull']].head(10)

    #age category (binned age)
    #df['AgeCategory']=math.floor((df['AgeFill'].astype(float) -  df['AgeFill'].astype(float) % 10)/10)
    df['AgeCategory']=(df['AgeFill'] // 10).astype(int)
    #print round(df['Age'][:5].astype(float)) % 10
    #print df['AgeCategory', 'AgeFill']
    print type(df['AgeFill'] // 10)

    #SibSp (siblings and/ or spouses), Parch (parent or child)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill*df.Pclass #these are not 

    return median_ages

def string2Enumeration(df):
    df['Sex_int'] = df['Sex'].map({j: i for i, j in enumerate(df['Sex'].dropna().unique())})
    df['Embarked_int'] = df['Embarked'].map({j: i for i, j in enumerate(df['Embarked'].dropna().unique())})

def fill_missing_fares(df, fares=None):

    #fill in ages based on Pclass and Sex information
    if fares==None:
        fares = np.zeros([2,3,10])

        for i in range(0,2):
            for j in range(0,3):
                for k in range(0,10):
                    #if len(df[(df['Sex_int']==i) & (df['Pclass']==j+1) & (df['AgeCategory']==k)]['Fare']):
                    fares[i,j,k]=df[(df['Sex_int']==i) & (df['Pclass']==j+1) & (df['AgeCategory']==k)]['Fare'].median()
                    #if math.isnan(fares[i,j,k]): fares[i,j,k] = 0
                    print i, j, k, fares[i,j,k]


    df['FareFill'] = df['Fare']

    #query first, and then fill
    print "Fare.isnull():"
    for (i,j,k) in df[df.Fare.isnull()][['Sex_int', 'Pclass', 'AgeCategory']].values:
        print i,j,k, fares[i,j-1,k]
        df.loc[(df.Fare.isnull()) & (df.Sex_int==i) & (df.Pclass==j) & (df['AgeCategory']==k), 'FareFill'] = fares[i,j-1,k]

    #    for i in range(0,2):
    #        for j in range(0,3):
    #            for k in range(0,10):
    #                df.loc[(df.Fare.isnull()) & (df.Sex_int==i) & (df.Pclass==j+1) & (df['AgeCategory']==k), 'FareFill'] = fares[i,j,k]
    #                if len(df[(df.Fare.isnull()) & (df.Sex_int==i) & (df.Pclass==j+1) & (df['AgeCategory']==k)]): print i,j,k, fares[i,j,k]

    return fares


def main():
    # For .read_csv, always use header=0 when you know row 0 is the header row
    train_df = pd.read_csv('../train.csv', header=0)
    test_df = pd.read_csv('../test.csv', header=0)
    
    #print "train_df:"
    #print train_df.dtypes
    #print train_df.describe()
    #print "test_df:"
    #print test_df.dtypes
    #print test_df.info()
    #print test_df.describe()

    print "train_df['Age'].isnull():"
    print len(train_df[train_df['Age'].isnull()])

    median_ages = fillMissingValues_NewVars(train_df)
    fillMissingValues_NewVars(test_df, median_ages)

    print "fares train_df:"
    fares = fill_missing_fares(train_df)
    print "fares test_df:"
    fill_missing_fares(test_df, fares)

    #transform distinct string values to enumerations:
    string2Enumeration(train_df)
    string2Enumeration(test_df)

    #print "train_df['AgeFill'], where train_df['Age']isnull():"
    #print train_df[train_df['Age'].isnull()]['AgeFill']
    #print "test_df.isnull():"
    #print test_df[test_df['Age'].isnull()][['Sex','Ticket']] #this makes all entries null for some reason, rather than printing those that are null
    #print "test_df.isnull().any(axis=1): ",type(test_df.isnull().any(axis=1))
    #print test_df[test_df.isnull().any(axis=1)][[i for i in test_df.columns]]
    #print len(test_df)
    #print [var in test_df.dtypes ]
    #print type(test_df.dtypes)
    #print test_df.columns
    #print type(test_df.columns) #this is an pandas index
    #print type(test_df.index) #a pandas Int64Index


    #delete string variables from dataframes
    drop_columns = ['Name','Sex', 'Ticket','Cabin','Embarked', 'Age', 'AgeFill', 'Fare', 'PassengerId']
    train_df=train_df.drop(drop_columns, axis=1)
    #print train_df.head(3)

    testPassengerIDs = test_df['PassengerId'].values
    test_df=test_df.drop(drop_columns, axis=1)
    #find PCA
    #apply PCA'd variables to train_df and test_df


    print "train_df.isnull():"
    #print train_df[train_df.isnull()]
    print len(train_df[train_df.isnull().any(axis=1)]) #evaluates to false for each column and row
    print "test_df.isnull():"
    print len(test_df[test_df.isnull().any(axis=1)])
    #print test_df[test_df.isnull().any(axis=1)][[i for i in test_df.columns]]


    #test_df = test_df[test_df.notnull().all(axis=1)]
    #print len(test_df[test_df.isnull().any(axis=1)])

    #make it usable for RandomForestClassifier, which expects a ndarray
    train_data = train_df.values #returns a numpy.ndarray
    test_data = test_df.values #returns a numpy.ndarray

    
    #sys.exit()

    print 'Training...'
    forest = RandomForestClassifier(n_estimators=10)
    forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
    
    print 'Predicting...'
    output = forest.predict(test_data).astype(int)
    
    outfilename = "my_randomforest.csv"
    predictions_file = open(outfilename, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(testPassengerIDs, output))
    predictions_file.close()
    print 'Done, created: ', outfilename



if __name__ == "__main__":
    main()
