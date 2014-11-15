from sklearn.covariance import GraphLassoCV, ledoit_wolf, GraphLasso, ShrunkCovariance
import sklearn.preprocessing as preprocessing
import pandas as pd


def precisionCol(cleandata, k):
    """Using precision matrix to choose useful attributes in high dimensional data"""
    model = ShrunkCovariance()
    model.fit(cleandata)
    pre_ = pd.DataFrame(model.get_precision())
    pre_.index = cleandata.columns
    pre_.columns = cleandata.columns
    test = abs(pre_['Y'])
    test.sort()
    test = test[-k:]
    coltest = (test.index).drop('Y')
    return coltest



def imputeNA(df):
    column = df.columns
    indexs = df.index
    impute = preprocessing.Imputer(strategy='most_frequent') #using most_frequent ones to fill in NA
    impute.fit(df)
    cleaned = impute.transform(df)
    cleanedData = pd.DataFrame(cleaned, columns=column, index = indexs)
    return cleanedData

