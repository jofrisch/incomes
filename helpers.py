import numpy as np
import pandas as pd
import re

from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
#from sklearn.metrics import accuracy_score, get_scorer, roc_auc_score

from sklearn.pipeline import Pipeline

#from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
#from sklearn.linear_model import LogisticRegression, LinearRegression
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


###############################
########## Selectors ##########
###############################

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #print(self.key, X[self.key].shape,type(X[self.key]) )
        return X[self.key]
    

class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #print(self.key, X[[self.key]].shape,type(X[[self.key]]))
        return X[[self.key]]


###############################
########### Encoders ##########
###############################


class DummyTransformer(BaseEstimator, TransformerMixin):
    """One Hot Encode a Pandas dataframe where all columns are categorical"""

    def __init__(self):
        self.dv = DictVectorizer(sparse=False)

    def fit(self, df, y=None):
        # assumes all columns of df are strings
        df_dict = df.to_dict('records')
        
        self.dv.fit(df_dict)
        return self

    def transform(self, df):
        assert isinstance(df, pd.DataFrame)

        df_dict = df.to_dict('records')
        df_t = self.dv.transform(df_dict)
        cols = self.dv.get_feature_names()
        df_dum = pd.DataFrame(df_t, index=df.index, columns=cols)
        return df_dum


class GetDummiesSeries(BaseEstimator, TransformerMixin):
    """One Hot Encode a Serie"""

    def __init__(self):
        self.dv = None

    def fit(self, serie, y=None):
        df_dict = pd.DataFrame(serie).to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(df_dict)
        return self

    def transform(self, serie):
        df_dict = pd.DataFrame(serie).to_dict('records')
        df_t = self.dv.transform(df_dict)
        cols = self.dv.get_feature_names()
        df_dum = pd.DataFrame(df_t, index=serie.index, columns=cols)
        return df_dum


class LabelEncodeByFreq(BaseEstimator, TransformerMixin):
    """ LabelEncode the categorical values of a Series X where 0 is the most frequent value, 1 the 2nd most frequent, etc. """

    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.freq_rank = {}
        for i, key in enumerate(X.value_counts().index):
            self.freq_rank[key]= i
        #when transforming we want to encode unseen values by a higher value than all the ones given until now
        self.nafilling = len(self.freq_rank)+1
        return self
    
    def transform(self, X):
        return pd.DataFrame(X.map(self.freq_rank).fillna(self.nafilling))



###############################
########### Imputers ##########
###############################

class TreatMissingsWithCommons(BaseEstimator, TransformerMixin):
    """
    Replace the missing values with 
    - the most frequent value in column for categories
    - mean values for numerics.
    """

    def __init__(self):
        self.replacements = pd.Series()

    def fit(self, X, y=None):

        assert isinstance(X, pd.DataFrame)

        col_relevant = []
        for col in X:
            if X[col].dtype=='object':
                col_relevant.append(X[col].mode()[0])
            else:
                col_relevant.append(X[col].mean())

        self.replacements = pd.Series(col_relevant, index=X.columns)
        return self.replacements

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        return X.fillna(self.replacements)


class KnnImputer(BaseEstimator, TransformerMixin):
    """Replaces the missing values of a DataFrame within one target variable, based on its k nearest neighbors identified with the other variables"""

    def __init__(self, target, n_neighbors = 41):
        self.target = target
        self.n_neighbors = n_neighbors
        self.miss = TreatMissingsWithCommons()
        self.ohe = DummyTransformer()
        
        
    def fit(self, X, y=None):
        try:
          #Replace numerical nans by mean() and categorical nans by the most frequent value
          self.miss.fit(X)
          X_full = self.miss.transform(X)

          #One Hot Encode categorical variables to pass the data to KNN
          self.ohe.fit(X_full)
          #Create a Dataframe that does not contain any nulls, categ variables are OHE, with all the rows of the original dataset that are not null
          X_ohe_full = self.ohe.transform(X_full[~X[self.target].isnull()].drop(self.target, axis=1))

          #Fit the classifier on lines where col is null
          if X[self.target].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
              self.knn = KNeighborsRegressor(n_neighbors = self.n_neighbors)
              self.knn.fit(X_ohe_full, X[self.target][~X[self.target].isnull()])
          else:
              self.knn = KNeighborsClassifier(n_neighbors = self.n_neighbors)
              self.knn.fit(X_ohe_full, X[self.target][~X[self.target].isnull()])

          return self
        
        except:
          print('Error while fitting KNNImputer, for target = ', self.target)
          raise
    
    def transform(self, X):
        try:
          #save index order
          self.locs = X.index.tolist() #save index, so that you can reconstruct the same df later
          X_full = self.miss.transform(X) #in transform
          #OHE on rows where col is null
          ohe_nulls = self.ohe.transform(X_full[X[self.target].isnull()].drop(self.target,axis=1))

          #Get prediction for nulls
          preds = self.knn.predict(ohe_nulls)

          ## Concatenate non nulls with nulls + target preds
          #Nulls + target preds
          X_nulls = X[X[self.target].isnull()].drop(self.target,axis=1)
          X_nulls[self.target] = preds

          X_imputed = pd.concat([X[~X[self.target].isnull()], X_nulls]).loc[self.locs]
          return X_imputed
        
        except:
          print('Error while transforming KNNImputer, for target = ', self.target)
          raise



###############################
########### Wrappers ##########
###############################

class ClassifierWrapper(BaseEstimator, TransformerMixin):
    """ Converts an estimator into a transformer."""
    
    def __init__(self, estimator, fit_params=None, use_proba=True, scoring=None):
        self.estimator = estimator
        self.fit_params= fit_params
        self.use_proba = use_proba #whether to use predict_proba in transform
        self.scoring = scoring # calculate validation score, takes score function name
        
        self.score = None #variable to keep the score if scoring is set.

    def fit(self,X,y):
        fp=self.fit_params
        
        if fp is not None:
            self.estimator.fit(X,y, **fp)
        else:
            self.estimator.fit(X,y)
        
        return self
    
    def transform(self, X):
        if self.use_proba:
            return self.estimator.predict_proba(X) #[:, 1].reshape(-1,1)
        else:
            return self.estimator.predict(X)
    
    def fit_transform(self,X,y,**kwargs):
        self.fit(X,y)
        p = self.transform(X)
        if self.scoring is not None:
            self.score = eval(self.scoring+"(y,p)")
        return p
    
    def predict(self,X):
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        return self.estimator.predict_proba(X)


class PipelineWrapperForEnsembling(BaseEstimator, TransformerMixin):
    """Converts a pipeline into a transformer. Normaly pipelines have already been fitted, if not, use pre_fitted_pipes = False, but this brings overfitting by construction"""
    
    def __init__(self, pipeline, use_proba=True, pre_fitted_pipes = True):
        self.pipeline = pipeline
        self.use_proba = use_proba #whether to use predict_proba in transform
        self.pre_fitted_pipes = pre_fitted_pipes

    def fit(self,X,y):
        if not self.pre_fitted_pipes:
            self.pipeline.fit(X,y)
        return self
    
    def transform(self, X):
        if self.use_proba:
            return self.pipeline.predict_proba(X)[:, 1].reshape(-1,1)
        else:
            return self.pipeline.predict(X)



###############################
######## GridSearching ########
###############################

def runGridSearch(paramGrid, pipe, X_train, y_train, verbose=2, n_jobs=-1, scoring='roc_auc'):
    """ Run a GridSearchCV for a given pipeline (pipe) on a parameter grid (paramGrid), and print results in an optimal way, following analyzeGridSearchResults()  """
    gs = GridSearchCV(pipe, paramGrid, verbose=verbose, n_jobs=n_jobs, scoring=scoring)
    gs.fit(X_train, y_train)
    analyzeGridSearchResults(gs, decimalsPrecision = 4)
    return gs

def analyzeGridSearchResults(gs, decimalsPrecision = 4):
    """ Simplify manual analysis of GridSearchCV results. """
    numResToAnalyze = int(len(gs.cv_results_['params'])*0.3) #show results of top 30% parameters sets
    mean_std = gs.cv_results_['mean_test_score'] - gs.cv_results_['std_test_score']
    bests = np.argsort(mean_std)[-numResToAnalyze:][::-1]
    for index in bests:
        print(gs.cv_results_['params'][index], " \t mean - std: ", round(mean_std[index], decimalsPrecision), " \t mean: ", round(gs.cv_results_['mean_test_score'][index], decimalsPrecision))