
## Introduction

We build a Machine Learning model to predict if a given adult's yearly income is above or below $50k.

We develop a Python package that implements a `get_pipeline` function that returns:

- [x] an [sklearn.pipeline.Pipeline](http://scikit-learn.org/stable/modules/pipeline.html)
- [x] that chains a series of [sklearn Transformers](http://scikit-learn.org/stable/data_transforms.html) to preprocess the data,
- [x] and ends with a [custom sklearn Estimator](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator) 
- [x] and is be fed a pandas DataFrame of the [Adult Data Set](http://mlr.cs.umass.edu/ml/datasets/Adult) to train and evaluate the pipeline.


## Getting started

1. Clone this repository 
2. Install [Miniconda](https://conda.io/miniconda.html)
3. Run `conda env create` from the repo's base directory to create the repo's conda environment from `environment.yml`. 
4. Run `activate machine-learning-challenge-env` to activate the conda environment.
5. Run `python main.py` to train the model and compute its ROC AUC between `y_pred` and `y_test`

## How it works

todo

## TO IMPROVE:
- improve get_pipeline() structure in pipelines.py
- train level1 pipelines on another dataset then level2 ensembling
- add a NN to the level1 models
