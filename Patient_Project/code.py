import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log
from sklearn import linear_model
from sklearn.cross_validation import *
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf, GraphLasso, ShrunkCovariance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVR
import sklearn.preprocessing as preprocessing
from supportfunction import *
from inferfunction import *


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
    cleandata = pd.read_csv("./data/cleaned_knnimpute.csv")
    cleandata.index = cleandata.sid
    cleandata = cleandata.drop('sid',1)
    mask = np.isnan(cleandata['Y'])
    cleandata = cleandata[mask == False]
    #cleandata = pd.read_csv("./data/cleaned.csv")
    #cleandata = cleandata.drop('Unnamed: 0', 1)
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
    cleandata = pd.read_csv("./data/cleaned_knnimpute.csv")
    cleandata.index = cleandata.sid
    cleandata = cleandata.drop('sid',1)
    #cleandata = pd.read_csv("./data/cleaned.csv")
    mask = np.isnan(cleandata['Y'])
    cleandata = cleandata[mask == False]
    #cleandata = cleandata.drop('Unnamed: 0', 1)
    #After c is chosen, use this to draw AUC plot
    train_id, test_id = train_test_split(cleandata.index, test_size=0.2)  # test_ratio = 0.2
    train = cleandata.ix[train_id]
    test = cleandata.ix[test_id]
    coltest = precisionCol(train, precisionk)
    coltest = list(coltest)
    coltest.append('Y')
    #train = train[coltest]
    #test = test[coltest]
    randomforest = RandomForestClassifier(n_estimators = 2000, n_jobs = -1, criterion = 'entropy', min_samples_split = min_samplesplit, min_samples_leaf = min_sampleleaf)
    randomforest.fit(train.drop('Y',1),train['Y'])
    fpr, tpr, thresholds = roc_curve(test['Y'], randomforest.predict_proba(test.drop('Y',1))[:,1])
    print auc(fpr, tpr)
    if draw == 'True':
        plotAUC(test['Y'], randomforest.predict_proba(test.drop('Y',1))[:,1], 'Random Forest')
        plt.savefig("testnorm_randomforest.png", dpi = 120)

def submissionrft(precisionk, min_samplesplit, min_sampleleaf):
    cleandata = pd.read_csv("./data/cleaned_knnimpute.csv")
    cleandata.index = cleandata.sid
    cleandata = cleandata.drop('sid',1)
    cleandata = cleandata[np.isnan(cleandata['Y']) == False]
    coltest = precisionCol(cleandata, precisionk)
    coltest = list(coltest)
    coltest.append('Y')
    data = pd.read_csv('./data/cleaned_knnimpute.csv')
    data.index = data.sid
    data.drop('sid',1)
    mask = np.isnan(data['Y'])
    train = data[mask == False]
    test = data[mask == True]
    #train = train[coltest]
    #test = test[coltest]
    indexs = test.index
    randomforest = RandomForestClassifier(n_estimators = 2000, n_jobs=-1, criterion = 'entropy', min_samples_split = min_samplesplit, min_samples_leaf = min_sampleleaf)
    randomforest.fit(train.drop('Y',1),train['Y'])
    ans = randomforest.predict_proba(test.drop('Y',1))[:,1]
    ans = pd.DataFrame(ans, index = indexs, columns=['Y'])
    submit = pd.read_csv('./data/CAX_ExacerbationModeling_SubmissionTemplate.csv')
    submit.index = submit['sid']
    submit = submit.drop('sid',1)
    submit['Exacebator'] = ans['Y']
    submit.to_csv('./data/CAX_ExacerbationModeling_Submission_rft_'+str(min_samplesplit)+'_'+str(min_sampleleaf)+'.csv')



def auctest(precisionk, c, draw = 'True'):
    #Read Data with dummy variables that has been cleaned and normalized
    cleandata = pd.read_csv("./data/cleaned_knnimpute.csv")
    cleandata.index = cleandata.sid
    cleandata = cleandata.drop('sid',1)
    mask = np.isnan(cleandata['Y'])
    cleandata = cleandata[mask == False]
    #cleandata = pd.read_csv("./data/cleaned.csv")
    #cleandata = cleandata.drop('Unnamed: 0', 1)
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


def othertest(precisionk, draw = 'False'):
    cleandata = pd.read_csv("./data/cleaned_knnimpute.csv")
    cleandata.index = cleandata.sid
    cleandata = cleandata.drop('sid',1)
    mask = np.isnan(cleandata['Y'])
    cleandata = cleandata[mask == False]
    #After c is chosen, use this to draw AUC plot
    train_id, test_id = train_test_split(cleandata.index, test_size=0.2)  # test_ratio = 0.2
    train = cleandata.ix[train_id]
    test = cleandata.ix[test_id]
    coltest = precisionCol(train, precisionk)
    coltest = list(coltest)
    coltest.append('Y')
    train = train[coltest]
    test = test[coltest]
    model = BaggingClassifier(base_estimator=linear_model.LogisticRegression(), n_estimators=100, max_features=200, n_jobs=-1)
    model.fit(train.drop('Y',1),train['Y'])
    fpr, tpr, thresholds = roc_curve(test['Y'], model.predict_proba(test.drop('Y',1))[:,1])
    print auc(fpr, tpr)
    if draw == 'True':
        plotAUC(test['Y'], model.decision_function(test.drop('Y',1)), 'Gradient Boosting')
        plt.savefig("testnorm_randomforest.png", dpi = 120)


def othersubmission(precisionk):
    cleandata = pd.read_csv("./data/cleaned_knnimpute.csv")
    cleandata.index = cleandata.sid
    cleandata = cleandata.drop('sid', 1)
    mask = np.isnan(cleandata['Y'])
    cleandata = cleandata[mask == False]
    coltest = precisionCol(cleandata, precisionk)
    coltest = list(coltest)
    coltest.append('Y')
    data = pd.read_csv('./data/cleaned_knnimpute.csv')
    data.index = data['sid']
    data.drop('sid',1)
    mask = np.isnan(data['Y'])
    train = data[mask == False]
    test = data[mask == True]
    train = train[coltest]
    test = test[coltest]
    indexs = test.index
    model = GradientBoostingClassifier()
    model.fit(train.drop('Y',1),train['Y'])
    ans = model.predict_proba(test.drop('Y',1))[:,1]
    ans = pd.DataFrame(ans, index = indexs, columns=['Y'])
    submit = pd.read_csv('./data/CAX_ExacerbationModeling_SubmissionTemplate.csv')
    submit.index = submit['sid']
    submit = submit.drop('sid',1)
    submit['Exacebator'] = ans['Y']
    submit.to_csv('./data/CAX_ExacerbationModeling_Submission_gbr'+str(precisionk)+'.csv')






def submissiontest(precisionk, c):
    #makeprediction
    #Read Data with dummy variables that has been cleaned and normalized
    cleandata = pd.read_csv("./data/cleaned_knnimpute.csv")
    cleandata.index = cleandata.sid
    cleandata = cleandata.drop('sid',1)
    mask = np.isnan(cleandata['Y'])
    cleandata = cleandata[mask == False]
    coltest = precisionCol(cleandata, precisionk)
    coltest = list(coltest)
    coltest.append('Y')
    data = pd.read_csv('./data/cleaned_knnimpute.csv')
    data.index = data['sid']
    data.drop('sid',1)
    mask = np.isnan(data['Y'])
    train = data[mask == False]
    test = data[mask == True]
    train = train[coltest]
    test = test[coltest]
    indexs = test.index
    logreg = linear_model.LogisticRegression(C=c)
    logreg.fit(train.drop('Y',1),train['Y'])
    ans = logreg.predict_proba(test.drop('Y',1))[:,1]
    ans = pd.DataFrame(ans, index = indexs, columns=['Y'])
    submit = pd.read_csv('./data/CAX_ExacerbationModeling_SubmissionTemplate.csv')
    submit.index = submit['sid']
    submit = submit.drop('sid',1)
    submit['Exacebator'] = ans['Y']
    submit.to_csv('./data/CAX_ExacerbationModeling_Submission_'+str(precisionk)+'_'+str(c).replace('.','')+'.csv')


def gbrtest(precisionk, min_sampleleaf, min_samplesplit, draw = 'False'):
    cleandata = pd.read_csv("./data/cleaned.csv")
    mask = np.isnan(cleandata['Y'])
    cleandata = cleandata[mask == False]
    cleandata = cleandata.drop('Unnamed: 0', 1)
    #After c is chosen, use this to draw AUC plot
    train_id, test_id = train_test_split(cleandata.index, test_size=0.2)  # test_ratio = 0.2
    train = cleandata.ix[train_id]
    test = cleandata.ix[test_id]
    coltest = precisionCol(train, precisionk)
    coltest = list(coltest)
    coltest.append('Y')
    #train = train[coltest]
    #test = test[coltest]
    gradientboost = GradientBoostingRegressor(n_estimators= 400, max_depth= 10, min_samples_split = min_samplesplit, min_samples_leaf = min_sampleleaf)
    gradientboost.fit(train.drop('Y',1),train['Y'])
    fpr, tpr, thresholds = roc_curve(test['Y'], gradientboost.decision_function(test.drop('Y',1)))
    print auc(fpr, tpr)
    if draw == 'True':
        plotAUC(test['Y'], gradientboost.decision_function(test.drop('Y',1)), 'Gradient Boosting')
        plt.savefig("testnorm_randomforest.png", dpi = 120)


def submissiongbr(precisionk, min_samplesplit, min_sampleleaf):
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
    #train = train[coltest]
    #test = test[coltest]
    indexs = test.index
    gradientboost = GradientBoostingRegressor(n_estimators= 400, max_depth= 10,min_samples_split = min_samplesplit, min_samples_leaf = min_sampleleaf)
    gradientboost.fit(train.drop('Y',1),train['Y'])
    ans = gradientboost.decision_function(test.drop('Y',1))
    ans = pd.DataFrame(ans, index = indexs, columns=['Y'])
    submit = pd.read_csv('./data/CAX_ExacerbationModeling_SubmissionTemplate.csv')
    submit.index = submit['sid']
    submit = submit.drop('sid',1)
    submit['Exacebator'] = ans['Y']
    submit.to_csv('./data/CAX_ExacerbationModeling_Submission_gbr_'+str(min_samplesplit)+'_'+str(min_sampleleaf)+'.csv')



def main():
    othertest(3050)
    #othersubmission(3050)
    #gbrtest(3000, min_sampleleaf=15, min_samplesplit=20)
    #submissiongbr(3000, min_sampleleaf=15, min_samplesplit=20)
    #rfttest(precisionk= 3050, min_sampleleaf=10, min_samplesplit=10)
    #submissionrft(precisionk=3050, min_sampleleaf=1, min_samplesplit=1)
    #auctest(3000, 0.0005, 'False')
    #submissiontest(3000, 0.0005)
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

