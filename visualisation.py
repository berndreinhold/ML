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
import pandas.tools.plotting as ptp
from pandas.compat import lrange
import seaborn as sns
import pandas.core.common as com

import html_output as ho

array_variables = ['Employment_Info', 'Family_Hist', 'Insurance_History', 'InsuredInfo', 'Medical_History', 'Medical_Keyword', 'Product_Info'] #list the variables with _[0-9]+ in order to count the number of nan per 'array' and to aggregate them

class fig_summary():
    def __init__(self):
        self.mean = -1
        self.RMS = -1
        self.xvar = ""
        self.yvar = ""
        self.label = ""
        self.comment = ""
        self.fig_path = "" #fig_path + fig_rel_path is where the figure is copied to physically
        self.fig_rel_path = "" #contains the relative path written to HTML and the name of the figure. It could be: figures/fig_name.png or just fig_name.png

class html_picture_summary_df(ho.html_picture_summary):
    #def __init__(self, of_name, CSS=[]):
    #    ho.html_picture_summary.__init__(self, of_name, CSS)

    def body_content(self, summary_struct, title, level=1, append=True): #level=1:<h1>, level=2:<h2>, level=3:<h3>
        l = []
        l[len(l):] = ["<h%d>%s</h%d>" % (level, title, level)]
        for s in summary_struct:
            l[len(l):] = ['<div class="Fig_BasicInfo">']
            l[len(l):] = ["<h%d>%s</h%d>" % (level+2, str(s.label), level+2)]
            l[len(l):] = ["<div>%s</div>" % str(s.fig_rel_path)]
            l[len(l):] = ['<img src="%s" alt="[no figure]"/>' % str(s.fig_rel_path)] #html code
            l[len(l):] = ["<div>%s</div>" % str(s.comment)]
            l[len(l):] = ['</div>']
        if not append: self._body_content = ""
        self._body_content += "\n".join(l)

 
class plot_summary():
    def __init__(self, df, html_dir="./", rel_dir=""):
        self._df = df #input dataframe from which to make plots
        if html_dir[-1]!="/": html_dir+="/"
        self._output_dir = html_dir #where the HTML file is being stored
        if len(rel_dir)>0 and rel_dir[-1]!="/": rel_dir+="/"
        self._rel_dir = rel_dir #the PNGs and the CSS are being stored in self._output_dir + self._rel_dir
        if not os.path.exists(self._output_dir + self._rel_dir): os.makedirs(self._output_dir + self._rel_dir)
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
        fs.label = "pair-wise correlation plots"
        fs.fig_path = self._output_dir
        fs.fig_rel_path = self._rel_dir + "correlation_plots_%s.png" % diagonal
        
        plt.savefig(fs.fig_path + fs.fig_rel_path)
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

    def corr_matrix(self, append=True, df=None, label=""):
        """
        produces correlation matrix
        """
        print("function: %s, label: %s, df is None: %s" % (sys._getframe().f_code.co_name, label, df is None))


        if df is None:
            corr_df = self._df.corr()
        else:
            corr_df = df.corr()

        for col in corr_df.columns:
            print(col)
        
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

        fig = plt.figure()
        #fig.autofmt_ydate() #http://matplotlib.org/1.3.1/users/recipes.html
        #plt.scatter(x,y, c=z, s=500, square=True, vmin=-1, vmax=1)
        #plt.heatmap(x,y, c=z, s=500, square=True, vmin=-1, vmax=1)
        #plt.scatter(x,y, c=z, s=500, vmax=1)
        #plt.scatter(x,y, c=z, s=500)
        #cb = plt.colorbar()
        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)

        #
        #sns.heatmap(corr_df, vmax=1, xticklabels=5, yticklabels=5, square=True) #plot only every 5th label, it 
        ax = sns.heatmap(corr_df, vmax=1, square=True, cbar=True)
        fig.autofmt_xdate(rotation=70) #does not exist for y-axis!
        ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0) #http://stackoverflow.com/questions/10998621/rotate-axis-text-in-python-matplotlib
        #fig.autofmt_ydate()
        
        #for i in range(len(x)):
        #    print(x[i], y[i], z[i])

        #    for y,label_y in enumerate(self._df.columns)
        
        fs = fig_summary()   #fs.mean = average(df[var_name])
        fs.label = "correlation matrix " + label
        fs.fig_path = self._output_dir
        if len(label)>0: label = "_" + label
        fs.fig_rel_path = self._rel_dir + "correlation_matrix%s.png" % label
        
        plt.savefig(fs.fig_path + fs.fig_rel_path)
        if not append: self.list_fig_summary.clear()
        self.list_fig_summary.append(fs)


    def perVar(self, var_name, nbins=10, bLog=False):
        """for each variable make a PNG and store some text"""
        print(self._df[var_name].dtype)
        fs = fig_summary()   #fs.mean = average(df[var_name])
        fs.xvar = var_name
        fs.label = var_name

        buffer = []
        if self._df[var_name].dtype in ["object"]: #variables to not plot
            buffer = self._df[var_name].dropna().unique()
            print(buffer[:10])
            fs.label = var_name
            fs.comment = "distinct values:\n"
            if len(buffer)<100:
                fs.comment += "\n".join(buffer)
            else:
                fs.comment += "\n".join(buffer[:100])
                fs.comment += "\n[...]"
        else:
            plt.figure()
            args = {}
            if bLog: args={'log': True}
            #args = {'bins': 120}
            self._df[var_name].hist(bins=nbins, **args)
            plt.xlabel(var_name)
            plt.suptitle = "%s" % var_name
            fs.xvar = var_name
            fs.yvar = "entries/bin"
            fs.label = var_name
            print(fs.label)
            fs.fig_path = self._output_dir
            if bLog: fs.fig_rel_path = self._rel_dir + var_name + "_log.png"
            else: fs.fig_rel_path = self._rel_dir + var_name + ".png"

        self.list_fig_summary.append(fs)

        #plt.show()
        print("figure made: ", fs.fig_path + fs.fig_rel_path)
        plt.savefig(fs.fig_path + fs.fig_rel_path)

    def plots_2D(self, range_padding=0.05):
        """
        make 2D correlation plots (hexbin) for numerical data
        strongly inspired by/ code copied from scatter_matrix() from pandas/tools/plotting.py
        plot normal and with log-z enabled next to each other.
        """

        df = self._df._get_numeric_data()
        n = df.columns.size
        naxes = n * n

        mask = com.notnull(df)

        boundaries_list = []
        for a in df.columns:
            values = df[a].values[mask[a].values]
            rmin_, rmax_ = np.min(values), np.max(values)
            rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
            boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))

        kwds = {'bins' : 'log'}

        for i, a in zip(lrange(n), df.columns):
            for j, b in zip(lrange(n), df.columns):
                if i == j: continue #only the non-diagonal elements

                fs = fig_summary()   #fs.mean = average(df[var_name])
                fig, axes = plt.subplots(1,2, sharey=True) #http://matplotlib.org/examples/pylab_examples/subplots_demo.html
                common = (mask[a] & mask[b]).values
                for ax in axes:
                    
                    ax.hexbin(df[b][common], df[a][common], gridsize=10, **kwds)
                    ax.set_xlim(boundaries_list[j])
                    ax.set_ylim(boundaries_list[i])
                    ax.set_xlabel(b)
                    ax.set_ylabel(a)

                fs.xvar = a
                fs.yvar = b
                fs.label = "%s_%s" % (a, b)
                #print(fs.label)
                fs.fig_path = self._output_dir
                fs.fig_rel_path = self._rel_dir + fs.label + ".png"

                self.list_fig_summary.append(fs)
                #plt.show()
                print("figure made: ", fs.fig_path + fs.fig_rel_path)
                plt.savefig(fs.fig_path + fs.fig_rel_path)

    def plots_2D_vs_response(self, range_padding=0.05):
        """
        make 2D correlation plots (hexbin) for numerical data vs the output variable 'response' (from Prudential kaggle competition)
        strongly inspired by/ code copied from scatter_matrix() from pandas/tools/plotting.py
        plot normal and with log-z enabled next to each other.
        """

        df = self._df._get_numeric_data()
        n = df.columns.size
        naxes = n * n

        mask = com.notnull(df)

        j = -1

        boundaries_list = []
        nbins_list = []
        for a in df.columns:
            values = df[a].values[mask[a].values]
            rmin_, rmax_ = np.min(values), np.max(values)
            rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
            boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))
            nbins = len(df[a].unique())
            if nbins>10: nbins_list.append(10)
            else: nbins_list.append(nbins)
            #print(nbins)
            if a == 'Response': j = len(boundaries_list)-1 #j is used below to access the boundaries_list variable

        if j<0: print("Error: Response-variable not found")

        kwds = {'bins' : 'log'}

        for i, a in zip(lrange(n), df.columns):
            if a == 'Response': continue
            elif a == 'Unnamed: 0': continue
            fs = fig_summary()   #fs.mean = average(df[var_name])
            #fig, axes = plt.subplots(1,2, sharey=True) #http://matplotlib.org/examples/pylab_examples/subplots_demo.html
            fig, axes = plt.subplots(1,2) #http://matplotlib.org/examples/pylab_examples/subplots_demo.html
            plt.subplots_adjust(wspace=0.4)
            common = (mask[a] & mask['Response']).values
            for k,ax in enumerate(axes):

                #cmap=plt.cm.YlOrRd_r
                if k==0: img = ax.hexbin(df['Response'][common], df[a][common], gridsize=(nbins_list[j],nbins_list[i]), cmap=plt.cm.Blues_r)
                else: img = ax.hexbin(df['Response'][common], df[a][common], gridsize=(nbins_list[j],nbins_list[i]),cmap=plt.cm.Blues_r, **kwds)
                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])
                ax.set_xlabel('Response')
                ax.set_ylabel(a)
                cb = plt.colorbar(img, ax=ax)
                if k==0: cb.set_label('entries')
                else: cb.set_label('log(entries)')

            fs.xvar = 'Response'
            fs.yvar = a
            fs.label = "%s_%s" % (a, 'Response')
            #print(fs.label)
            fs.fig_path = self._output_dir
            fs.fig_rel_path = self._rel_dir + fs.label + ".png"
            
            self.list_fig_summary.append(fs)
            #plt.show()
            print("figure made: ", fs.fig_path + fs.fig_rel_path)
            plt.savefig(fs.fig_path + fs.fig_rel_path)

    def __del__(self):
        print(self.list_fig_summary)


def main():
    # For .read_csv, always use header=0 when you know row 0 is the header row
    #train_df = pd.read_csv('/home/reinhold/data/ML/input_data/train.csv', header=0)
    #df = pd.read_csv('/home/reinhold/data/ML/Prudential/input_data/train_Prudential.csv', header=0)
    df = pd.read_csv('/home/reinhold/data/ML/Prudential/intermediate_data/train_Prudential_cleaned.csv', header=0)
    #df = pd.read_csv('/home/reinhold/data/ML/intermediate_data/train_df_cleaned.csv', header=0)

    output_filepath = "/home/reinhold/data/ML/Prudential/intermediate_data/"
    output_filename = "train_Prudential_beforeAnything.html"
    #output_filename = "train_Prudential_afterCleaning.html"
    #output_filename = "train_Prudential_corr_matrices.html"
    #output_filename = "TitanicTrainingSet_AfterCleanUp.html"
    html = html_picture_summary_df(output_filename, output_filepath, ["basic_style.css"])

    plot_sum = plot_summary(df, output_filepath, "figures/")
    #plot_sum = plot_summary(train_df, output_filepath)

    if 0:
        html.body_content([], "correlation matrix and correlation plots (columns pair-wise)", 2, False)
        #plot_sum.corr_plots(False) #scatter_matrix quite powerful, ignores string variables automatically

        in_ = []
        out_ = [] #list of columns, that are excluded from the usual correlation matrix: all array_variables, except their aggregates (var_count_null, var_sum)
        for col in df.columns:
            if col.find("count_null")!=-1 or col.find("sum")!=-1:
                in_[len(in_):] = [col]
            for av in array_variables:
                if col.startswith(av):
                    out_[len(out_):] = [col]

        all_set = set(df.columns)
        in_set = set(in_)
        out_set = set(out_)

        all_columns = all_set - (out_set - in_set) - set(["Unnamed: 0"]) #double exclusion, yeah! first out_set - in_set: exclude count_null, sum from array_variables, because we want to keep these in the overall dataset.
        print(all_columns)

        column_list = [x for x in all_columns]
        column_list.sort()

        temp_df = df[column_list]
        label = "all_except_array_variables"
        plot_sum.corr_matrix(False, temp_df, label) 
        html.body_content(plot_sum.list_fig_summary, "correlation matrix (%s)" % label, 3, True)

        for av in array_variables:
            temp_df = df.filter(regex=av)
            label = av
            plot_sum.corr_matrix(False, temp_df, label) 
            html.body_content(plot_sum.list_fig_summary, "correlation matrix (%s)" % label, 3, True)

    if 1:

        #plot_sum.list_fig_summary.clear()
        for vars in df.columns: 
            print(vars)
            nbins = len(df[vars].unique())
            if nbins>10: nbins=10
            plot_sum.perVar(vars, nbins, True) #includes also non-numeric columns
        html.body_content(plot_sum.list_fig_summary, "1D plots of each variable", 2, True)


        ##pairwise plots only for numeric columns --- too much, don't use
        #plot_sum.list_fig_summary.clear()
        #plot_sum.plots_2D()
        #html.body_content(plot_sum.list_fig_summary, "pair-wise correlation plots of two numeric variables", 2, True)

    if 0:
        ##pairwise plots only for numeric columns
        plot_sum.list_fig_summary.clear()
        plot_sum.plots_2D_vs_response()
        html.body_content(plot_sum.list_fig_summary, "pair-wise correlation plots of each variable with the 'Response'", 2, True)


    html.loop("Prudential Training Set (after cleaning)")


if __name__ == "__main__":
    status = main()
    sys.exit(status)
