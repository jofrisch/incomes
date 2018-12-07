import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


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
  
  def __init__(self, position):
    self.position = position
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    print("####################")
    print("Data Quality for ", self.position)
    print("####################")
    print("DataFrame/Series contains any nulls:", X['native-country'].isnull().any())
    return X


class CheckNpArrayQuality(BaseEstimator, TransformerMixin):
  
  def __init__(self, position):
    self.position = position
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    print("####################")
    print("Data Quality for ", self.position)
    print("####################")
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
    print("####################")
    print("Print lines with nulls:")
    print(X[np.isnan(X).any(axis=1)][:3])
    print("####################")
    print('First 5 lines:')
    print(X[:5])
    return X


