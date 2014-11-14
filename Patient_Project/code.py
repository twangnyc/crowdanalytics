import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import *
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf, GraphLasso, ShrunkCovariance
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as preprocessing





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




def precisionCol(cleandata, k):
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


def getNumericCol():
    '''Get column having numerical content'''
    document = "./data/CAX_ExacerbationModeling_MetaData.csv"
    coldata = pd.read_csv(document)
    col = coldata[coldata['Column Type'] == 'Numeric']['varnum']
    return col

def getCategoricalCol():
    '''Get column having categorical content'''
    document = "./data/CAX_ExacerbationModeling_MetaData.csv"
    coldata = pd.read_csv(document)
    col = coldata[coldata['Column Type'] == 'Category']['varnum']
    return col


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
        if float(na)/length >= 0.4 :
            cleandf = cleandf.drop(column, 1)   #Drop data columns with too many NA
    print "Delete %d columns" %(len(df.columns) - len(cleandf.columns))
    return cleandf


def imputeNA(df):
    column = df.columns
    indexs = df.index
    impute = preprocessing.Imputer(strategy='most_frequent') #using most_frequent ones to fill in NA
    impute.fit(df)
    cleaned = impute.transform(df)
    cleanedData = pd.DataFrame(cleaned, columns=column, index = indexs)
    return cleanedData


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


def submissiontest(precisionk, c):
    if False:
    #Create and clean the big data
        cleandata = pd.read_csv("./data/cleaned.csv")
        cleandata = cleandata.drop('Unnamed: 0', 1)
        coltest = precisionCol(cleandata, 200)
        coltest = list(coltest)
        coltest.append('Y')

        document = "./data/CAX_ExacerbationModeling_TRAIN_data.csv"
        data = pd.read_csv(document)
        testdocument = "./data/CAX_ExacerbationModeling_Public_TEST_data.csv"
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
        cleandata = imputeNA(cleandata)
        cleandata.index = index_
        cleandata = dummyData(cleandata)
        cleandata.index = index_
        cleandata = normData(cleandata)
        cleandata.index = index_

    if True:
        #makeprediction
        #Read Data with dummy variables that has been cleaned and normalized
        cleandata = pd.read_csv("./data/cleaned.csv")
        cleandata = cleandata.drop('Unnamed: 0', 1)
        coltest = precisionCol(cleandata, precisionk)
        coltest = list(coltest)
        coltest.append('Y')

        data = pd.read_csv('./data/bigcleaned.csv')
        data.index = data['sid']
        data.drop('sid',1)
        mask = np.isnan(data['Y'])
        train = data[mask == False]
        test = data[mask == True]
        train = train[coltest]
        test = test[coltest]
        indexs = test.index
        logreg = linear_model.LogisticRegression(penalty='l1', C=c)
        logreg.fit(train.drop('Y',1),train['Y'])
        ans = logreg.predict_proba(test.drop('Y',1))[:,1]
        ans = pd.DataFrame(ans, index = indexs, columns=['Y'])
        submit = pd.read_csv('./data/CAX_ExacerbationModeling_SubmissionTemplate.csv')
        submit.index = submit['sid']
        submit = submit.drop('sid',1)
        submit['Exacebator'] = ans['Y']
        submit.to_csv('./data/CAX_ExacerbationModeling_Submission_'+str(precisionk)+'_'+str(c)+'.csv')


def xValAUC(tr, lab, k, cs, precision):
    """
    Perform k-fold cross validation on logistic regression, varies C,
    returns a dictionary where key=c,value=[auc-c1, auc-c2, ...auc-ck].
    """
    cv = KFold(n=tr.shape[0], n_folds = k)
    aucs = {}

    for train_index, test_index in cv:
        tr_f = tr.iloc[train_index]
        va_f = tr.iloc[test_index]
        coltest = precisionCol(tr_f, precision)
        coltest = list(coltest)
        coltest.append(lab)
        tr_f = tr_f[coltest]
        va_f = va_f[coltest]
        for c in cs:
            logreg = linear_model.LogisticRegression(penalty='l1', C=c)
            logreg.fit(tr_f.drop(lab,1),tr_f[lab])
            met = roc_auc_score(va_f[lab], logreg.predict_proba(va_f.drop(lab,1))[:,1])

            if (aucs.has_key(c)):
                aucs[c].append(met)
            else:
                aucs[c] = [met]

    return aucs

def auctest(precisionk, c, draw = 'True'):
    #Read Data with dummy variables that has been cleaned and normalized
    cleandata = pd.read_csv("./data/cleaned.csv")
    cleandata = cleandata.drop('Unnamed: 0', 1)
    #After c is chosen, use this to draw AUC plot
    train_id, test_id = train_test_split(cleandata.index, test_size=0.2)  # test_ratio = 0.3
    train = cleandata.ix[train_id]
    coltest = precisionCol(train, precisionk)
    coltest = list(coltest)
    coltest.append('Y')
    train = train[coltest]
    test = cleandata.ix[test_id]
    test = test[coltest]
    logreg = linear_model.LogisticRegression(C=c)
    logreg.fit(train.drop('Y',1),train['Y'])
    fpr, tpr, thresholds = roc_curve(test['Y'], logreg.predict_proba(test.drop('Y',1))[:,1])
    print auc(fpr, tpr)
    if draw == 'True':
        plotAUC(test['Y'], logreg.predict_proba(test.drop('Y',1))[:,1], 'LR')
        plt.savefig("testnorm_"+str(precisionk)+"_"+str(c)+".png", dpi = 120)


def ctest(topk):
    #Read Data with dummy variables that has been cleaned and normalized
    cleandata = pd.read_csv("./data/cleaned.csv")
    cleandata = cleandata.drop('Unnamed: 0', 1)

    xval_dict = {'e':[], 'mu':[], 'sig':[]}
    cs = [10**i for i in range(-10,10)]
    auc_cv = xValAUC(cleandata, 'Y', 5, cs, topk)
    for i in range(-10,10):
        xval_dict['e'].append(i)
        xval_dict['mu'].append(np.array(auc_cv[10**i]).mean())
        xval_dict['sig'].append(np.sqrt(np.array(auc_cv[10**i]).var()))
    res = pd.DataFrame(xval_dict)
    res['low'] = res['mu'] - res['sig']/sqrt(len(cs))
    res['up'] = res['mu'] + res['sig']/sqrt(len(cs))
    plt.figure()
    plt.plot(res['e'], res['mu'])
    plt.plot(res['e'], res['low'], 'k--')
    plt.plot(res['e'], res['up'], 'k--')
    plt.legend(loc=4)
    plt.xlabel('log10 of C')
    plt.ylabel('Mean Val AUC')
    plt.title('X-validated AUC by C')
    plt.savefig("CV_C"+str(topk)+".png", dpi = 120)

def rfttest(precisionk, min_samplesplit, min_sampleleaf, draw = 'True'):
    cleandata = pd.read_csv("./data/cleaned.csv")
    cleandata = cleandata.drop('Unnamed: 0', 1)
    #After c is chosen, use this to draw AUC plot
    train_id, test_id = train_test_split(cleandata.index, test_size=0.2)  # test_ratio = 0.2
    train = cleandata.ix[train_id]
    test = cleandata.ix[test_id]
    coltest = precisionCol(train, precisionk)
    coltest = list(coltest)
    coltest.append('Y')
    train = train[coltest]
    test = test[coltest]
    randomforest = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', min_samples_split = min_samplesplit, min_samples_leaf = min_sampleleaf)
    randomforest.fit(train.drop('Y',1),train['Y'])
    fpr, tpr, thresholds = roc_curve(test['Y'], randomforest.predict_proba(test.drop('Y',1))[:,1])
    print auc(fpr, tpr)
    if draw == 'True':
        plotAUC(test['Y'], randomforest.predict_proba(test.drop('Y',1))[:,1], 'Random Forest')
        plt.savefig("testnorm_randomforest.png", dpi = 120)

def submissionrft(precisionk, min_samplesplit, min_sampleleaf):
    cleandata = pd.read_csv("./data/cleaned.csv")
    cleandata = cleandata.drop('Unnamed: 0', 1)
    coltest = precisionCol(cleandata, precisionk)
    coltest = list(coltest)
    coltest.append('Y')
    data = pd.read_csv('./data/bigcleaned.csv')
    data.index = data['sid']
    data.drop('sid',1)
    mask = np.isnan(data['Y'])
    train = data[mask == False]
    test = data[mask == True]
    train = train[coltest]
    test = test[coltest]
    indexs = test.index
    randomforest = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', min_samples_split = min_samplesplit, min_samples_leaf = min_sampleleaf)
    randomforest.fit(train.drop('Y',1),train['Y'])
    ans = randomforest.predict_proba(test.drop('Y',1))[:,1]
    ans = pd.DataFrame(ans, index = indexs, columns=['Y'])
    submit = pd.read_csv('./data/CAX_ExacerbationModeling_SubmissionTemplate.csv')
    submit.index = submit['sid']
    submit = submit.drop('sid',1)
    submit['Exacebator'] = ans['Y']
    submit.to_csv('./data/CAX_ExacerbationModeling_Submission_rft_'+str(min_samplesplit)+'_'+str(min_sampleleaf)+'.csv')



def main():
    rfttest(precisionk= 500, min_sampleleaf=50, min_samplesplit=50)
    submissionrft(precisionk=500, min_sampleleaf=50, min_samplesplit=50)
    #auctest(50, 0.1, 'False')
    #submissiontest(500, 0.005)
    #for c in range(1,1001,200):
    #    ctest(c)


    if False:
        #Clean Data, Create Dummy variable and Normalize
        document = "./data/CAX_ExacerbationModeling_TRAIN_data.csv"
        data = pd.read_csv(document)
        data = data.drop('sid', 1)
        data['Y'] = data['Exacebator']
        data = data.drop('Exacebator', 1)
        cleandata = dropNAdata(data)  # drop column having too much NA
        cleandata = imputeNA(cleandata)  # replace NA by most frequent content
        cleandata = dummyData(cleandata)  # create dummy variables for categorical data
        cleandata = normData(cleandata)  # normalize data
        coltest = precisionCol(cleandata, 200)
        coltest = list(coltest)
        coltest.append('Y')

    if False:
        #Read Data with dummy variables that has been cleaned and normalized
        cleandata = pd.read_csv("./data/cleaned.csv")
        cleandata = cleandata.drop('Unnamed: 0', 1)
        coltest = precisionCol(cleandata, 200)
        coltest = list(coltest)
        coltest.append('Y')

    if False:
        #CV to choose C
        xval_dict = {'e':[], 'mu':[], 'sig':[]}
        cs = [10**i for i in range(-10,10)]
        auc_cv = xValAUC(cleandata, 'Y', 5, cs, 500)
        for i in range(-10,10):
            xval_dict['e'].append(i)
            xval_dict['mu'].append(np.array(auc_cv[10**i]).mean())
            xval_dict['sig'].append(np.sqrt(np.array(auc_cv[10**i]).var()))
        res = pd.DataFrame(xval_dict)
        res['low'] = res['mu'] - res['sig']/sqrt(len(cs))
        res['up'] = res['mu'] + res['sig']/sqrt(len(cs))
        plt.figure()
        plt.plot(res['e'], res['mu'])
        plt.plot(res['e'], res['low'], 'k--')
        plt.plot(res['e'], res['up'], 'k--')
        plt.legend(loc=4)
        plt.xlabel('log10 of C')
        plt.ylabel('Mean Val AUC')
        plt.title('X-validated AUC by C')
        plt.savefig("CV_C.png", dpi = 120)


    if False:
        #After c is chosen, use this to draw AUC plot
        train_id, test_id = train_test_split(range(0, len(cleandata)), test_size=0.2)  # test_ratio = 0.3
        train = cleandata.ix[train_id]
        coltest = precisionCol(train, 50)
        coltest = list(coltest)
        coltest.append('Y')
        train = train[coltest]
        test = cleandata.ix[test_id]
        test = test[coltest]
        logreg = linear_model.LogisticRegression(C=0.01)
        logreg.fit(train.drop('Y',1),train['Y'])
        plotAUC(test['Y'], logreg.predict_proba(test.drop('Y',1))[:,1], 'LR')
        fpr, tpr, thresholds = roc_curve(test['Y'], logreg.predict_proba(test.drop('Y',1))[:,1])
        print auc(fpr, tpr)
        plt.savefig("testnorm.png", dpi = 120)

    if False:
        #REAL STUFF!!!
        #Clean Data, Create Dummy variable and Normalize
        document = "./data/CAX_ExacerbationModeling_TRAIN_data.csv"
        data = pd.read_csv(document)
        data = data.drop('sid', 1)
        data['Y'] = data['Exacebator']
        data = data.drop('Exacebator', 1)
        data = dropNAdata(data)  # drop column having too much NA
        remain_column = data.columns
        remain_column = remain_column.drop('Y')
        #Compute remain_column
        testdocument = "./data/CAX_ExacerbationModeling_Public_TEST_data.csv"
        test = pd.read_csv(testdocument)
        test = test.drop('sid', 1)
        test = test.drop('Exacebator', 1)
        test = test[remain_column]
        cleantest = dummyData(test)  # create dummy variables for categorical data
        cleantest = normData(cleantest)
        cleantest = cleantest[coltest.remove('Y')]





if __name__ == "__main__":
    main()

