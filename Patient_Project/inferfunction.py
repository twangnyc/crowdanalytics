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


def fullData(df):
    full_index = df.isnull().sum(1)==0
    full_index = full_index[full_index == True].index
    return df.ix[full_index]


def simpleimputeNA(df, method = 'mean'):
    """
    This function imputes the columns having less than 45 missing values with mean value.
    It's a preparation to do graphical lasso and knn imputation.
    """
    na_limit = (df.isnull().sum(0) < 45)
    simple_col = na_limit[na_limit==True].index
    indexs = df.index
    simple_df = df[simple_col]
    impute = preprocessing.Imputer(strategy=method) #using most_frequent ones to fill in NA
    impute.fit(simple_df)
    simple_cleaned = impute.transform(simple_df)
    simple_cleaned = pd.DataFrame(simple_cleaned, columns = simple_col, index = indexs)
    cleanedData = df[:]
    cleanedData[simple_col] = simple_cleaned[simple_col]
    return cleanedData

