"""
adjusted from https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii
Author : Bernd
Date : Dec. 30, 2015
"""

print(__doc__)
import pandas as pd
import numpy as np
import pylab as p
import sys

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../train.csv', header=0)

#print df #a lot of stuff
#print df.head(3)
#print df.tail(3)
#print type(df)
#print df.dtypes
print df.info()
#print df.describe() #only shown for float64 and int64 columns, not object columns
#print df['Age'][:10]
#print df.Age[:10]
#print type(df.Age)
#print df.Age.mean(), df.Age.std(), df.Age.median()

#print df[df.Age>60][['Age','Sex','Pclass', 'Survived']] #column name has to match exactly, otherwise an error is raised

#for i in range(1,4):
#    print i, len(df[(df.Sex == 'male') & (df.Pclass ==i)])

#Age histogram
#df.Age.hist(range=(0,80), bins=16)
#p.show()

def test(x):
    if x[0].upper()=='F': return 0
    elif x[0].upper()=='M': return 1
    else: return None


#cleaning up data:
#df['Sex_int'] = df['Sex'].map(lambda x: x[0].upper())
#df['Sex_int'] = df['Sex'].str.upper().map({'FEMALE':0, 'MALE':1}) #works
#df['Sex_int'] = df['Sex'].str.upper().map({'F':0, 'M':1}) #does not work
df['Sex_int'] = df['Sex'].map(test).astype(int) #works
#df['Sex_int'] = df['Sex'].map(lambda x: x[0].upper() and {'F':0, 'M':1}).astype(int) #pipe'ing does not quite work, tried, based on processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s) on http://www.diveintopython.net/power_of_introspection/lambda_functions.html
#df['Sex_int'] = df.pipe(lambda x: x[0].upper(), arg1='Sex').map({'FEMALE':0, 'MALE':1}).astype(int) #did not work
#print help(df.Embarked.dropna().mode())
print df.Embarked.dropna().mode().values #simply returns one value - don't quite understand why (ToDo?)
print df.Sex_int.mode().values

#port_index, port_name = enumerate(df.Embarked.unique())
ports_dict = {ports: i for i, ports in enumerate(df.Embarked.unique())}
print ports_dict

ports_dict = {ports: i for i, ports in enumerate(df.Embarked.dropna().unique())}
print ports_dict

#df['Embarked_int'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2}) # (Cherbourg, Southamption and Queenstown), from https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests
sys.exit()

#print df[['Sex', 'Sex_int', 'Embarked', 'Embarked_int']][:10]
#print df.describe()
#pass

#df.Sex_int.hist()
#p.show()
#df.Embarked_int.hist()
#p.show()

#fill in ages based on Pclass and Sex information

median_ages = np.zeros([2,3])

for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j]=df[(df['Sex_int']==i) & (df['Pclass']==j+1)]['Age'].median()
        #median_ages[i,j]=i*j

print "median age per Sex and Pclass:"
print median_ages

df['AgeFill'] = df['Age']
#print df[df['Age'].isnull()][['Sex_int','Pclass', 'Age', 'AgeFill']].head(10)

for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Age.isnull()) & (df.Sex_int==i) & (df.Pclass==j+1), 'AgeFill'] = median_ages[i,j]


df['AgeIsNull']=df['Age'].isnull().astype(int)

#print df[df['Age'].isnull()][['Sex_int','Pclass', 'Age', 'AgeFill', 'AgeIsNull']].head(10)

#SibSp (siblings and/ or spouses), Parch (parent or child)
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill*df.Pclass #these are not 

#list only object variables
#print df.dtypes
#print df.dtypes[df.dtypes.map(lambda x: x=='object')]
#print df.dtypes[df.dtypes.map(lambda x: x!='object')]

print len(df)

#remove unused columns:
df = df.drop(['Name','Sex', 'Ticket','Cabin','Embarked', 'Age'], axis=1)
#df.drop(['Name'], axis=0) 
df = df.dropna() #drop

print df.dtypes
print len(df)

print type(df[df['Fare']<20]['Fare'].hist())
#p.show()

train_data = df.values #returns a numpy.ndarray
#print train_data
#print type(train_data)

#list distinct values - extremely useful
#print df['Sex'].unique()

#        try:
#            row[8] = float(row[8])    # No fare recorded will come up as a string so
                                      # try to make it a float
#        except:                       # If fails then just bin the fare according to the class
#            bin_fare = 3 - float(row[1])
#            break                     # Break from the loop and move to the next row
#        if row[8] > fare_ceiling:     # Otherwise now test to see if it is higher
#                                      # than the fare ceiling we set earlier
#            bin_fare = number_of_price_brackets - 1
#            break                     # And then break to the next row
#
#        if row[8] >= j*fare_bracket_size\
#            and row[8] < (j+1)*fare_bracket_size:     # If passed these tests then loop through
#                                                      # each bin until you find the right one
#                                                      # append it to the bin_fare
#                                                      # and move to the next loop
#            bin_fare = j
#            break
        # Now I have the binned fare, passenger class, and whether female or male, we can
        # just cross ref their details with our survival table
#    if row[3] == 'female':
#        predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, bin_fare ])])
#    else:
#        predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, bin_fare])])

# Close out the files
#test_file.close()
#predictions_file.close()
