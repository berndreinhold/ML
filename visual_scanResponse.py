"""
produce visualisations of a data frame
Author : Bernd
Date : Jan. 6, 2015
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import matplotlib

matplotlib.style.use('ggplot')



def main():
    # For .read_csv, always use header=0 when you know row 0 is the header row
    #train_df = pd.read_csv('/home/reinhold/data/ML/input_data/train.csv', header=0)
    #df = pd.read_csv('/home/reinhold/data/ML/Prudential/input_data/train_Prudential.csv', header=0)
    df = pd.read_csv('/home/reinhold/data/ML/Prudential/intermediate_data/train_PCA_variance_ratio.csv', header=0)
    #df = pd.read_csv('/home/reinhold/data/ML/intermediate_data/train_df_cleaned.csv', header=0)

    df = df.cumsum()
    df['Number of Components'] = pd.Series(list(range(len(df))))

    ax = df[['Number of Components', 'response_1']].plot(x='Number of Components', y='response_1',linestyle='-')
    for r in range(2,9):
        df[['Number of Components', 'response_%d'%r]].plot(x='Number of Components', y='response_%d' %r, linestyle='-', ax=ax)

    plt.show()
    #plt.savefig("test.png")



if __name__ == "__main__":
    status = main()
    sys.exit(status)
