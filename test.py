#!/usr/bin/env Python
#try out examples from http://pandas.pydata.org/pandas-docs/stable/visualization.html

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#mpl.style.use('ggplot')
#pd.options.display.mpl_style = 'default'
#plt.hold(True)

x = ['A']*300 + ['B']*400 + ['C']*300
y = np.random.randn(1000)
df = DataFrame({'Letter':x, 'N':y})
grouped = df.groupby('Letter')



#print df[['Letter','N']][290:310]

#df['N'].hist(by=df['Letter'])

#print df[[x for x in df.columns]]

#df = df.reset_index().pivot('index','Letter','N')
#df = df.reset_index()

#plt.savefig("test2.png")


#print df[[x for x in df.columns]]


#ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

#print ts
#ts = ts.cumsum()
#ts.plot()

#plt.show()


#df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
#df = df.cumsum()
#plt.figure();
#df.plot();

#df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
#df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C'])
#df3['A'] = pd.Series(list(range(len(df))))

#print df3
#df3.plot(x='C', y='B')
#plt.show()


df4 = pd.DataFrame({'a': np.random.randn(1000) + 2, 'b': np.random.randn(1000),
                    'c': np.random.randn(1000) - 2}, columns=['a', 'b', 'c'])

#fig = plt.figure()

#df4.plot.hist
df4.plot(kind='hist', alpha=0.5)
plt.show()
#plt.savefig('test.png')
