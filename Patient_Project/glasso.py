import pandas as pd
from sklearn.covariance import GraphLassoCV, ledoit_wolf, GraphLasso, ShrunkCovariance
import sklearn.preprocessing as preprocessing


def precisionCol(cleandata, k):
    model = GraphLasso(mode = 'lars')
    model.fit(cleandata)
    pre_ = pd.DataFrame(model.get_precision())
    pre_.index = cleandata.columns
    pre_.columns = cleandata.columns
    pre_.to_csv("precision.csv")
    test = abs(pre_['Y'])
    test.sort()
    test = test[-k:]
    coltest = (test.index).drop('Y')
    return coltest

def main():
    cleandata = pd.read_csv("./data/cleaned.csv")
    cleandata = cleandata.drop('Unnamed: 0', 1)
    coltest = precisionCol(cleandata, 200)
    print coltest

if __name__ == '__main__':
    main()




