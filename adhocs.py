import pandas as pd
import numpy as np

import os

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


###########################
##### Feature creation ####
###########################

class AdHocFeaturesCreation(BaseEstimator, TransformerMixin):
    """ Create features that are specific to the problem's dataset."""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.df = X
        self.df["is-marital-status-favo"] = self.df["marital-status"].apply(lambda x: int(x in ['Married-civ-spouse','Married-AF-spouse']))
        self.df["is-education-favo"] = self.df["education"].apply(lambda x: int('th' in x or 'Preschool' in x))
        self.df["capital-diff"] = self.df["capital-gain"] - self.df["capital-loss"]
        self.df['is-married'] = self.df.apply(lambda row: int(row['relationship'] in ['Husband','Wife'] or "Married" in row["marital-status"]), axis=1) 
        return self.df


##############################
###### Ad Hoc Checks #########
##############################

class CheckDFQuality(BaseEstimator, TransformerMixin):
  """ Performs adhoc checks on a dataframe, within a pipeline"""
  def __init__(self, position):
    self.position = position
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    print("#"*20)
    print("Data Quality for ", self.position)
    print("#"*20)
    print("DataFrame/Series contains any nulls:", X['native-country'].isnull().any())
    return X


class CheckNpArrayQuality(BaseEstimator, TransformerMixin):
  """ Performs adhoc checks on a numpy array, within a pipeline"""  
  def __init__(self, position):
    self.position = position
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    print("#"*20)
    print("Data Quality for ", self.position)
    print("#"*20)
    print('Contains nulls:')
    print(np.any(np.isnan(X)))
    print("Is finite:")
    print(np.all(np.isfinite(X)))
    print('Max:')
    print(X.flatten().max())
    print('Shape:')
    print(X.shape)
    print('Nb of unique lines:')
    print(len(np.unique(X, axis=0)))
    print("#"*20)
    print("Print lines with nulls:")
    print(X[np.isnan(X).any(axis=1)][:3])
    print("#"*20)
    print('First 5 lines:')
    print(X[:5])
    return X


#########################
#### Download data ######
#########################


def download_data(subset='train'):

    fname = os.path.join(os.getcwd(), "data", "adult_"+ subset + ".csv")

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