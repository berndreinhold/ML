"""
clean up the dataset, fill missing values, transform string to integer values and drop string variables, bin the ages
this is very specific to the dataset itself
Author : Bernd
Date : Jan. 13, 2015
"""

print(__doc__)
import pandas as pd
import numpy as np
import sys
import csv

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

        print("median age per Sex and Pclass:")
        print(median_ages)

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
    print(type(df['AgeFill'] // 10))

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
                    #print i, j, k, fares[i,j,k]


    df['FareFill'] = df['Fare']

    #query first, and then fill
    print("Fare.isnull():")
    for (i,j,k) in df[df.Fare.isnull()][['Sex_int', 'Pclass', 'AgeCategory']].values:
        #print i,j,k, fares[i,j-1,k]
        df.loc[(df.Fare.isnull()) & (df.Sex_int==i) & (df.Pclass==j) & (df['AgeCategory']==k), 'FareFill'] = fares[i,j-1,k]

    #    for i in range(0,2):
    #        for j in range(0,3):
    #            for k in range(0,10):
    #                df.loc[(df.Fare.isnull()) & (df.Sex_int==i) & (df.Pclass==j+1) & (df['AgeCategory']==k), 'FareFill'] = fares[i,j,k]
    #                if len(df[(df.Fare.isnull()) & (df.Sex_int==i) & (df.Pclass==j+1) & (df['AgeCategory']==k)]): print i,j,k, fares[i,j,k]

    return fares


def main():
    input_dir = '/home/reinhold/data/ML/input_data/'
    
    median_ages = None #will be filled in the training data set
    fares = None #will be filled in the training data set
    
    dataset_names = ["train", "cv", "test"]
    output_dir = '/home/reinhold/data/ML/intermediate_data/'

    for name in dataset_names:
        # For .read_csv, always use header=0 when you know row 0 is the header row
        df = pd.read_csv('%s/%s.csv' % (input_dir, name), header=0)
        
        #print "%s:" % name
        #print df.dtypes
        #print df.info
        #print df.describe()

        if median_ages is None: median_ages = fillMissingValues_NewVars(df)
        else: fillMissingValues_NewVars(df, median_ages)

        print("fares %s:" % name)
        if fares is None:
            fares = fill_missing_fares(df)
        else:
            fill_missing_fares(df, fares)            

        #transform distinct string values to enumerations:
        string2Enumeration(df)

        #delete string variables from dataframes
        drop_columns = ['Name','Sex', 'Ticket','Cabin','Embarked', 'Age', 'AgeFill', 'Fare']
        #keep PasssengerId for later -> drop at the very last minute

        #drop columns,
        #check for null-values (there should be none!)


        df=df.drop(drop_columns, axis=1)
        if len(df[df.isnull().any(axis=1)]): print("WARNING: %s.isnull(): %d" % (name, len(df[df.isnull().any(axis=1)]))) #evaluates to false for each column and row

        #store in output file
        outfilename = "%s/%s_df_cleaned.csv" % (output_dir, name)
        outfile = open(outfilename, "wb")
        open_file_object = csv.writer(outfile)
        print([c for c in df.columns])
        print([df.columns]) #df.columns is of type Index
        open_file_object.writerow([c for c in df.columns])
        open_file_object.writerows(df.values)
        outfile.close()
        print("Done, created: %s" % outfilename)



if __name__ == "__main__":
    main()


#keep these for reference
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


