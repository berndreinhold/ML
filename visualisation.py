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
import pandas.tools.plotting as ptp
from pandas.compat import lrange
import seaborn as sns

import html_output as ho

class fig_summary():
    def __init__(self):
        self.mean = -1
        self.RMS = -1
        self.xvar = ""
        self.yvar = ""
        self.label = ""
        self.comment = ""
        self.fig_path = ""

class html_picture_summary_df(ho.html_picture_summary):
    #def __init__(self, of_name, CSS=[]):
    #    ho.html_picture_summary.__init__(self, of_name, CSS)

    def body_content(self, summary_struct, title, level=1, append=True): #level=1:<h1>, level=2:<h2>, level=3:<h3>
        l = []
        l[len(l):] = ["<h%d>%s</h%d>" % (level, title, level)]
        for s in summary_struct:
            l[len(l):] = ['<img src="%s" alt="%s"/>' % (str(s.fig_path), str(s.label))] #html code
            l[len(l):] = ["%s, %s" % (str(s.fig_path), str(s.label))]

        if not append: self._body_content = ""
        self._body_content += "\n".join(l)


class plot_summary():
    def __init__(self, df, outdir="./"):
        self._df = df #input dataframe from which to make plots
        if outdir[-1]!="/": outdir+="/"
        self._output_dir = outdir #where the PNGs are being produced
        self.list_fig_summary = []

    def corr_plots(self, append=True):
        """
        uses scatter_matrix to plot all columns pair-wise against each other 
        """
        plt.figure()
        #self._df.drop([x for ])
        #ptp.scatter_matrix(self._df, alpha=0.2, figsize=(6, 6), diagonal='kde')
        #ptp.scatter_matrix(self._df, alpha=0.2, diagonal='kde')
        #diagonal='kde'
        #ptp.scatter_matrix(self._df, alpha=0.2, diagonal=diagonal)
        #print(diagonal)
        #plt.savefig(self._output_dir + "correlation_matrix_%s.png" % diagonal)

        diagonal='hist'
        print(sys._getframe().f_code.co_name, diagonal)
        #scatter_opt={'kind':'hexbin'}
        off_diagonal_opt={'bins':'log'} #draw options for off-diagonal elements
        #scatter_opt={}
        #scatter_matrix(self._df, alpha=0.2, hspace=0.2, wspace=0.2, diagonal=diagonal, **scatter_opt)
        #axes = ptp.scatter_matrix(self._df, alpha=0.2, diagonal=diagonal, **scatter_opt)
        ptp.scatter_matrix(self._df, alpha=0.2, diagonal=diagonal, **off_diagonal_opt)
        #scatter_matrix(self._df, alpha=0.2, diagonal=diagonal)

        fs = fig_summary()   #fs.mean = average(df[var_name])
        fs.label = "correlation matrix"
        fs.fig_path = self._output_dir + "correlation_plots_%s.png" % diagonal
        
        plt.savefig(fs.fig_path)
        if not append: self.list_fig_summary.clear()
        self.list_fig_summary.append(fs)


        #n=self._df.columns.size
        #for i, a in zip(lrange(n), self._df.columns):
        #    for j, b in zip(lrange(n), self._df.columns):
        #        ax = axes[i, j]
        #        print(ax)
        #        plt.figure()
        #        ax.plot()
        #        print("correlation_matrix_%s_%s.png" % (ax.get_xlabel(), ax.get_ylabel()))
        #        plt.savefig(self._output_dir + "correlation_matrix_%s_%s.png" % (ax.get_xlabel(), ax.get_ylabel()))

    def corr_matrix(self, append=True):
        """
        produces correlation matrix
        """
        print(sys._getframe().f_code.co_name)
        plt.figure()
        corr_df = self._df.corr()
        #print("corr_df: ")
        #print(corr_df)
        #print(corr_df.info())
        #print(corr_df.describe())
        #print ("len(corr_df): %d" % len(corr_df))
        #corr_df.plot(, kind='scatter')
        
        #x, y, z = [], [], []
        #buffer = []
        #var_labels = []
        #buffer = range(corr_df.columns.size)
        #for i,label in enumerate(corr_df.columns):
        #    for y_i, corr_xy in enumerate(corr_df[label].values):
        #        x.append(i)
        #        y.append(y_i)
        #        z.append(corr_xy)

        #plt.scatter(x,y, c=z, s=500, square=True, vmin=-1, vmax=1)
        #plt.heatmap(x,y, c=z, s=500, square=True, vmin=-1, vmax=1)
        #plt.scatter(x,y, c=z, s=500, vmax=1)
        #plt.scatter(x,y, c=z, s=500)
        #cb = plt.colorbar()
        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)

        #
        #sns.heatmap(corr_df, vmax=1, xticklabels=5, yticklabels=5, square=True) #plot only every 5th label, it 
        sns.heatmap(corr_df, vmax=1, square=True)

        #for i in range(len(x)):
        #    print(x[i], y[i], z[i])

        #    for y,label_y in enumerate(self._df.columns)
        
        fs = fig_summary()   #fs.mean = average(df[var_name])
        fs.label = "correlation matrix"
        fs.fig_path = self._output_dir + "correlation_matrix.png"
        
        plt.savefig(fs.fig_path)
        if not append: self.list_fig_summary.clear()
        self.list_fig_summary.append(fs)


    def perVar(self, var_name):
        """for each variable make a PNG and store some text"""
        print(self._df[var_name].dtype)
        fs = fig_summary()   #fs.mean = average(df[var_name])
        fs.xvar = var_name
        fs.label = var_name

        buffer = []
        if self._df[var_name].dtype in ["object"]: #variables to not plot
            buffer = self._df[var_name].dropna().unique()
            print(buffer[:10])
            if len(buffer)<100:
                fs.label = "\n".join(buffer)
            else:
                fs.label = "\n".join(buffer[:100])
        else:
            plt.figure()
            #args={'log': True}
            #args = {'bins': 120}
            self._df[var_name].hist()
            plt.xlabel(var_name)
            plt.suptitle = "%s" % var_name
            fs.xvar = var_name
            fs.yvar = "entries/bin"
            fs.label = var_name
            print(fs.label)
            fs.fig_path = self._output_dir + var_name + ".png"

        self.list_fig_summary.append(fs)

        #plt.show()
        print("figure made: ", fs.fig_path)
        plt.savefig(fs.fig_path)

    def __del__(self):
        print(self.list_fig_summary)


def main():
    # For .read_csv, always use header=0 when you know row 0 is the header row
    train_df = pd.read_csv('/home/reinhold/data/ML/input_data/train.csv', header=0)

    output_file = "/home/reinhold/data/ML/output_data/mytest.html"
    h = html_picture_summary_df(output_file, ["Nachmieter.css"])

    x = plot_summary(train_df, "/home/reinhold/data/ML/output_data/")

    x.corr_plots(False) #scatter_matrix quite powerful, ignores string variables automatically
    x.corr_matrix(True) 
    h.body_content(x.list_fig_summary, "correlation matrix and correlation plots (columns pair-wise)", 2, True)

    x.list_fig_summary.clear()

    #print(train_df.info())
    #print type(train_df.columns)
    for vars in  train_df.columns:
        print(vars)
        x.perVar(vars)
    h.body_content(x.list_fig_summary, "1D plots of each variable", 2, True)

    h.loop("Titanic Training Set")

    
if __name__ == "__main__":
    status = main()
    sys.exit(status)
