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
from pandas.tools.plotting import scatter_matrix

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

    def body_content(self, summary_struct, title, level=1): #level=1:<h1>, level=2:<h2>, level=3:<h3>
        l = []
        l[len(l):] = ["<h%d>%s</h%d>" % (level, title, level)]
        for s in summary_struct:
            l[len(l):] = ['<img src="%s" alt="%s"/>' % (str(s.fig_path), str(s.label))] #html code
            l[len(l):] = ["%s, %s" % (str(s.fig_path), str(s.label))]

        self._body_content = "\n".join(l)


class plot_1D():
    def __init__(self, df, outdir="./"):
        self._df = df #input dataframe from which to make plots
        if outdir[-1]!="/": outdir+="/"
        self._output_dir = outdir #where the PNGs are being produced
        self.list_fig_summary = []

    def corr_plots(self):
        plt.figure()
        #self._df.drop([x for ])
        #scatter_matrix(self._df, alpha=0.2, figsize=(6, 6), diagonal='kde')
        #scatter_matrix(self._df, alpha=0.2, diagonal='kde')
        diagonal='kde'
        scatter_matrix(self._df, alpha=0.2, diagonal=diagonal)
        print(diagonal)
        plt.savefig(self._output_dir + "correlation_matrix_%s.png" % diagonal)

        diagonal='hist'
        scatter_opt={'kind':'hexbin', 'color':'red'}
        #scatter_opt={'color':'red'}
        scatter_matrix(self._df, alpha=0.2, hspace=0.2, wspace=0.2, diagonal=diagonal, **scatter_opt)
        print(diagonal)
        plt.savefig(self._output_dir + "correlation_matrix_%s.png" % diagonal)

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

    x = plot_1D(train_df, "/home/reinhold/data/ML/output_data/")

    print(train_df.info())
    #print type(train_df.columns)
    #for vars in  train_df.columns:
    #    print vars
    #    x.perVar(vars)

    x.corr_plots() #scatter_matrix quite powerful, ignores string variables automatically

    h.body_content(x.list_fig_summary, "1D plots of each variable", 2)
    h.loop("Titanic Training Set")

    
if __name__ == "__main__":
    status = main()
    sys.exit(status)
