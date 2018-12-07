import numpy as np
import pandas as pd
import re


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, get_scorer, roc_auc_score


from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from adhocs import *
from helpers import *

###########################################################
########### Our main pipeline will follow this logic:
########### - We create ad-hoc new features, that were defined during the EDA (exploratory data analysis)
########### - The pipeline is then forked in 2 pipelines:
###########     1. The first one (called p1) aims to generate a dataframe with a limited number of features (that are expected to be the most relevant ones by EDA. We then run 2 complex algorithms (Random Forest and SVM()
###########     2. The second one (called p2) aims to generate a dataframe with a larger number of features (without limiting the number of features and using One Hot Encoding instead of label encoding for the first pipeline). We then run 2 simpler algorithms (Logistic Regression with Ridge and  Gradient Boosting Trees)
########### - The results of these 4 algorithms are ensembled through a Linear Regression to get the final prediction
##########################################################


##########################################################
#create one pipeline for each variable
#for race and native-country we create 2 piplines as they will be used differently in the 2 sub-piplines
#other variables are used similarly or used only in one pipeline
##########################################################



age = Pipeline([
                ('selector', NumberSelector(key='age')),
                ('standard', StandardScaler())
            ])

fnlwgt = Pipeline([
                ('selector', NumberSelector(key='fnlwgt')),
                ('standard', StandardScaler())
            ])

education_num = Pipeline([
                ('selector', NumberSelector(key='education-num')),
                ('standard', StandardScaler())
            ])

capital_gain = Pipeline([
                ('selector', NumberSelector(key='capital-gain')),
                ('standard', StandardScaler())
            ])

capital_loss = Pipeline([
                ('selector', NumberSelector(key='capital-loss')),
                ('standard', StandardScaler())
            ])

sex = Pipeline([
                ('selector', TextSelector(key='sex')),
                ('freq', LabelEncodeByFreq())
            ])


hours_per_week = Pipeline([
                ('selector', NumberSelector(key='hours-per-week')),
                ('standard', StandardScaler())
            ])


capital_diff = Pipeline([
                ('selector', NumberSelector(key='capital-diff')),
                ('standard', StandardScaler())
            ])

is_marital_status_favo = Pipeline([
                ('selector', NumberSelector(key='is-marital-status-favo'))
            ])

is_education_favo = Pipeline([
                ('selector', NumberSelector(key='is-education-favo'))
            ])

is_married = Pipeline([
                ('selector', NumberSelector(key='is-married'))
            ])

race1 = Pipeline([
                ('selector', TextSelector(key='race')),
                ('freq', LabelEncodeByFreq())
            ])

race2 = Pipeline([
                ('selector', TextSelector(key='race')),
                ('ohe', GetDummiesSeries())
            ])

native_country1 = Pipeline([
                ('selector', TextSelector(key='native-country')),
                ('freq', LabelEncodeByFreq())
            ])

native_country2 = Pipeline([
                ('selector', TextSelector(key='native-country')),
                ('ohe', GetDummiesSeries())
            ])

workclass = Pipeline([
                ('selector', TextSelector(key='workclass')),
                ('ohe', GetDummiesSeries())
            ])

occupation = Pipeline([
                ('selector', TextSelector(key='occupation')),
                ('ohe', GetDummiesSeries())
            ])

relationship = Pipeline([
                ('selector', TextSelector(key='relationship')),
                ('ohe', GetDummiesSeries())
            ])



###################################
##### Feature Selection ###########
###################################

feats1 = FeatureUnion([('age', age),
                      ('education_num', education_num),
                      ('is_education_favo', is_education_favo),
                      ('is_marital_status_favo', is_marital_status_favo),
                      ('hours_per_week', hours_per_week),
                      ('capital_diff', capital_diff),
                      ('sex', sex),
                      ('race', race1),
                      ('native_country', native_country1),
                      ('is_married', is_married)
                     ])

feats2 = FeatureUnion([('age', age),
                      ('education_num', education_num),
                      ('is_education_favo', is_education_favo),
                      ('is_marital_status_favo', is_marital_status_favo),
                      ('hours_per_week', hours_per_week),
                      ('capital_diff', capital_diff),
                      ('capital_gain', capital_gain),
                      ('capital_loss', capital_loss),
                      ('sex', sex),
                      ('race', race2),
                      ('native_country', native_country2),
                      ('occupation', occupation),
                      ('relationship', relationship),
                      ('workclass', workclass),
                      ('is_married', is_married)
                     ])


####################################
##### All level 1 pipelines ########
####################################


pipe11 = Pipeline([
        
    
        # Imputing missing values
        ('imputers1', KnnImputer(target = 'native-country')),
    
        #DataQuality
        #('quality1', CheckDFQuality(position='Post Imputer')),
    
        # Create extra features
        ('adhocFC',AdHocFeaturesCreation()),
    
        #DataQuality
        #('quality2', CheckDFQuality(position='Post Feat Creation')),
    
        #Feature Selection
        ('feats1', feats1),
    
        #DataQuality
        #('quality3', CheckNpArrayQuality(position='Post Feat Selection')),

        #Training model
        ('classifier', RandomForestClassifier(bootstrap = True, n_estimators = 100, max_depth=10, max_features='auto'))
        ])


pipe12 = Pipeline([
    
        # Imputing missing values
        ('imputers1', KnnImputer(target = 'native-country', n_neighbors=41)),
    
        # Create extra features
        ('adhocFC',AdHocFeaturesCreation()),
    
        #DataQuality
        #('quality2', CheckDFQuality(position='Post Feat Creation')),

        #Feature Selection
        ('feats1', feats1),
    
        #DataQuality
        #('quality3', CheckNpArrayQuality(position='Post Feat Selection')),

        #Training model
        ('classifier', SVC(C = 10, gamma = 0.1, kernel = 'rbf', probability=True))
        ])



pipe21 = Pipeline([
    
        # Imputing missing values
        ('imputers1', KnnImputer(target = 'native-country', n_neighbors=41)),
        ('imputers2', KnnImputer(target = 'workclass', n_neighbors=41)),
        ('imputers3', KnnImputer(target = 'occupation', n_neighbors=41)),
    
        # Create extra features
        ('adhocFC',AdHocFeaturesCreation()),
    
        #DataQuality
        #('quality2', CheckDFQuality(position='Post Feat Creation')),

        #Feature Selection
        ('feats2', feats2),
    
        #DataQuality
        #('quality3', CheckNpArrayQuality(position='Post Feat Selection')),

        #Training model
        ('classifier', LogisticRegression(C = 0.3, penalty='l2'))
        ])


pipe22 = Pipeline([
    
        # Imputing missing values
        ('imputers1', KnnImputer(target = 'native-country', n_neighbors=41)),
        ('imputers2', KnnImputer(target = 'workclass', n_neighbors=41)),
        ('imputers3', KnnImputer(target = 'occupation', n_neighbors=41)),
    
        # Create extra features
        ('adhocFC',AdHocFeaturesCreation()),
    
        #DataQuality
        #('quality2', CheckDFQuality(position='Post Feat Creation')),

        #Feature Selection
        ('feats2', feats2),
    
        #DataQuality
        #('quality3', CheckNpArrayQuality(position='Post Feat Selection')),

        #Training model
        ('classifier', GradientBoostingClassifier(loss='deviance', max_depth=3, n_estimators=600))
        ])




#Define hyperparameter grids
hyperparameters_11 = {'classifier__bootstrap':[True, False], 'classifier__max_depth':[3,10,30], 'classifier__max_features':['auto','sqrt'], 'classifier__n_estimators':[30,100,300,600]}
hyperparameters_12 = {'classifier__C':[0.1, 0.3, 1, 3, 10], 'classifier__gamma':[0.01,0.03, 0.1, 0.3], 'classifier__kernel':['linear','rbf']}
hyperparameters_21 = {'classifier__C':[0.1, 0.3, 1, 3, 10], 'classifier__penalty':['l1','l2']}
hyperparameters_22 = {'classifier__loss':['deviance','exponential'], 'classifier__n_estimators':[30,100,300,600], 'classifier__max_depth':[3,10,30]}


##################################
###### All Level 2 pipelines #####
##################################

stacking_pipelines = FeatureUnion([
    ('pipe11', PipelineWrapperForEnsembling(pipe11, use_proba=True, pre_fitted_pipes= False)),
    ('pipe12', PipelineWrapperForEnsembling(pipe12, use_proba=True, pre_fitted_pipes= False)),
    ('pipe21', PipelineWrapperForEnsembling(pipe21, use_proba=True, pre_fitted_pipes= False)),
    ('pipe22', PipelineWrapperForEnsembling(pipe22, use_proba=True, pre_fitted_pipes= False))
])


ensembling = Pipeline([
    #Stack the predictions of each basic model 
    ('stacking', stacking_pipelines),
    
    #Run meta Model
    ('meta', LogisticRegression(penalty = 'l1'))
])



def get_pipeline():
	return ensembling



if __name__ == '__main__':
    """ Fine tune each pipeline, separately."""
    from main import download_data, rs
    from helpers import runGridSearch, runGridSearch
    import sys
    from sklearn.model_selection import train_test_split



    pipe_to_tune = sys.argv[1]
    hyperparameters_dict = {'11':hyperparameters_11, '12':hyperparameters_12, '21':hyperparameters_21, '22':hyperparameters_22}
    pipelines_dict={'11': pipe11, '12':pipe12, '21':pipe21, '22':pipe22}

    #Download and structure the data
    X_, y_ = download_data(subset='train')
    X_test, y_test = download_data(subset='test')
    X_train, X_meta, y_train, y_meta = train_test_split(X_, y_, test_size = 0.2, random_state = rs)

    print("Data has been downloaded!")

    runGridSearch(hyperparameters_dict[pipe_to_tune], pipelines_dict[pipe_to_tune], X_train, y_train, verbose=1, n_jobs=-1, scoring='roc_auc')

