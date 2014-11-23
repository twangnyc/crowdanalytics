import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import NearestNeighbors
from supportfunction import *
from inferfunction import *


def knnimpute(df):
    full = df.ix[df.isnull().sum(1) == 0]
    df_na = df.ix[df.isnull().sum(1) != 0]
    na_index = df_na.index
    df_impute = df[:]
    for i in range(len(na_index)):
        print float(i)/len(na_index)
        na_col = list(df_na.ix[na_index[i]][df_na.ix[na_index[i]].isnull()].index)
        full_col = list(df_na.ix[na_index[i]][df_na.ix[na_index[i]].notnull()].index)
        knn_train = full[full_col]
        model = NearestNeighbors(n_neighbors=1)
        model.fit(knn_train)
        neighbour = model.kneighbors(df_na.ix[na_index[i]][full_col], return_distance=False)[0][0]
        df_impute.ix[na_index[i]][na_col] = full.ix[knn_train.iloc[neighbour].name][na_col]
    return df_impute


def cleandata():
    document = "/Users/Tian/Documents/NYU/CrowdAnalytics/Patient_Project/data/CAX_ExacerbationModeling_TRAIN_data.csv"
    data = pd.read_csv(document)
    testdocument = "/Users/Tian/Documents/NYU/CrowdAnalytics/Patient_Project/data/CAX_ExacerbationModeling_Public_TEST_data.csv"
    test = pd.read_csv(testdocument)
    data = data.append(test)
    data.index = data['sid']
    data = data.drop('sid',1)
    data = data.sort_index()
    Y = data['Exacebator']
    Y = pd.DataFrame(Y, index=data.index)
    data = data.drop('Exacebator',1)
    index_ = data.index
    cleandata = dropNAdata(data)
    cleandata.index = index_
    cleandata = simpleimputeNA(cleandata) # using mean value to impute columns with small amount of missing data
    cleandata.index = index_
    cleandata = knnimpute(cleandata)
    cleandata = dummyData(cleandata)
    cleandata.index = index_
    cleandata = normData(cleandata)
    cleandata.index = index_
    cleandata['Y'] = Y
    cleandata.to_csv('/Users/Tian/Documents/NYU/CrowdAnalytics/Patient_Project/data/cleaned_knnimpute.csv')

if __name__ == '__main__':
    cleandata()