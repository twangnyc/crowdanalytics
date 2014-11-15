import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from supportfunction import *


def cleandata_noimpute():
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
    cleandata = cleandata.fillna(-999)
    cleandata = dummyData(cleandata)
    cleandata.index = index_
    cleandata = normData(cleandata)
    cleandata.index = index_
    cleandata['Y'] = Y
    cleandata.to_csv('/Users/Tian/Documents/NYU/CrowdAnalytics/Patient_Project/data/cleaned_noimpute.csv')

if __name__ == '__main__':
    cleandata_noimpute()