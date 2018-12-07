import numpy as np
import pandas as pd
import re
from sklearn.model_selection import GridSearchCV, train_test_split
from pipelines import get_pipeline
from sklearn.metrics import roc_auc_score
import sys
import os.path

rs = 123


def download_data(subset='train'):

    fname = "adult_"+ subset + ".csv"

    if not os.path.isfile(fname) :
        # Construct the data URL.
        if subset == 'train':
            csv_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data'
        else:
            csv_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test'
        # Define the column names.
        names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'earns_over_50K']
        # Read the CSV.
        print('Downloading {subset} dataset to __data__/ ...')
        df = pd.read_csv(
            csv_url,
            sep=', ',
            names=names,
            skiprows=int(subset == 'test'),
            na_values='?')

        # Split into feature matrix X and labels y.
        df.earns_over_50K = df.earns_over_50K.str.contains('>').astype(int)
        df.to_csv(fname, index=False)
        print("Data has been downloaded, and copied to local disk")

    else:
        df = pd.read_csv(fname)
        print("Data has been copied from local disk")

    X, y = df.drop(['earns_over_50K'], axis=1), df.earns_over_50K
    return X, y



if __name__ == '__main__':

    #Download and structure the data
    X_train, y_train = download_data(subset='train')
    X_test, y_test = download_data(subset='test')

    

    #get the full pipeline
    pipeline = get_pipeline()

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(roc_auc_score(y_test, y_pred))