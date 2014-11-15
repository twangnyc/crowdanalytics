import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc


def getDfSummary(df):
    length=len(df)
    column=df.columns
    summary=df.describe().T
    length2=len(summary)
    summary['NA Count']=length-summary['count']
    diffnum=[1]*length2
    j=0
    for i in column:
        diffnum[j]=len(df[i].value_counts())
        j=j+1
    summary['distinct values']=diffnum
    return summary


def getNAnumber(df):
    length = len(df)
    column = df.columns
    summary = df.describe().T
    list = length - summary['count']
    return list



def dropNAdata(df):
    columns = df.columns
    cleandf = df[:]
    length = len(df)
    for column in columns:
        na = getNAnumber(pd.DataFrame(df[column]))
        if float(na)/length >= 0.2 :
            cleandf = cleandf.drop(column, 1)  # Drop data columns with too many NA
    print "Delete %d columns" % (len(df.columns) - len(cleandf.columns))
    return cleandf

def getNumericCol():
    '''Get column having numerical content'''
    document = "/Users/Tian/Documents/NYU/CrowdAnalytics/Patient_Project/data/CAX_ExacerbationModeling_MetaData.csv"
    coldata = pd.read_csv(document)
    col = coldata[coldata['Column Type'] == 'Numeric']['varnum']
    return col


def getCategoricalCol():
    '''Get column having categorical content'''
    document = "/Users/Tian/Documents/NYU/CrowdAnalytics/Patient_Project/data/CAX_ExacerbationModeling_MetaData.csv"
    coldata = pd.read_csv(document)
    col = coldata[coldata['Column Type'] == 'Category']['varnum']
    return col


def normData(df):
    data = df[:]
    numCol = getNumericCol() # only normalize numerical data
    try:
        for col in numCol:
            data[col] = preprocessing.scale(data[col])
    except:
        pass
    return data

def plotAUC(truth, pred, lab):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color=c, label= lab+' (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")


def dummyData(df):
    """This function make categorical data to dummy variable"""
    data = df[:]
    cg_column = list(getCategoricalCol())
    df_column = list(df.columns)
    column = list(np.intersect1d(cg_column, df_column))
    for col in column:
        print col
        n_value = len(data[col].value_counts())
        if n_value > 2:
            dummy = pd.get_dummies(data[col], col)
            data = data.join(dummy)
            data = data.drop(col, 1)
    print "Create %d dummy variables" % (len(data.columns)-len(df.columns))
    return data
