import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.utils import create_userids, print_list
from util.normalization import normalize_rows
from sklearn import metrics
import util.settings as st
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def plot_scores(positive_scores, negative_scores, filename='scores.png', title='Score distribution'):
    set_style()
    plt.clf()
    df = pd.DataFrame([positive_scores, negative_scores])
    BINS = np.linspace(df.min(), df.max(), 31)
    sns.distplot(positive_scores, norm_hist=True, color='green', bins=31)
    sns.distplot(negative_scores, norm_hist=True, color='red', bins=31)
    # plt.legend(loc='upper left')
    plt.legend(['Genuine', 'Impostor'], loc='best')
    plt.xlabel('Score')
    plt.title(title)
    plt.show()
    # plt.savefig(filename + '.png')

    

def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper", font_scale = 2)
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    sns.set_style("ticks")
    sns.set_style("whitegrid")


def plot_ROC_single(ee_file, title = 'ROC curve'):
    set_style()
    ee_data  = pd.read_csv(ee_file) 
    auc_ee = metrics.auc(ee_data['FPR'], ee_data['TPR'])

    plt.clf()
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.plot(ee_data['FPR'], ee_data['TPR'], '-', label = 'AUC_EE = %0.2f' % auc_ee)

    label_ee = 'AUC  = %0.2f' % auc_ee
    legend_str = [label_ee]
    plt.legend(legend_str)
    plt.show()
   

# create a boxplot from a dataframe
# 
def csv2boxplot(df, columns, title, ylabel, outputfilename):
    myFig = plt.figure()
    res = df.boxplot(column=columns, return_type='axes')
    plt.title(title)
    plt.xlabel('Type of features')
    plt.ylabel(ylabel)
    myFig.savefig('output_png/boxplot_sapimouse.png', format = 'png')
    myFig.savefig(outputfilename + '.png', format='png')
    # myFig.savefig(outputfilename + '.eps', format='eps')
    # plt.show(res)

def plot_ROC_filelist(filelist, title = 'ROC curve', outputfilename='roc.png'):
    set_style()
    plt.clf()
    counter = 1
    labels = []
    for file in filelist:
        data = pd.read_csv(file) 
        auc = metrics.auc(data['FPR'], data['TPR'])
        # label = 'blocks: %2d' % counter
        label = 'AUC #blocks:%2d = %0.2f' % (counter, auc)
        plt.plot(data['FPR'], data['TPR'], label = 'AUC %2d blocks = %0.2f' % (counter, auc))
        labels.append( label )
        counter = counter + 1
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(labels)
    plt.show()
    # plt.savefig(outputfilename + '.png', format='png')
    # plt.savefig(outputfilename + '.eps', format='eps')