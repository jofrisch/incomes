import numpy as np
import pandas as pd
import re

from pipelines import get_pipeline
from adhocs import download_data
from sklearn.metrics import roc_auc_score


rs = 123


if __name__ == '__main__':

    #Download and structure the data
    X_train, y_train = download_data(subset='train')
    X_test, y_test = download_data(subset='test')

    

    #get the full pipeline
    pipeline = get_pipeline()

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(roc_auc_score(y_test, y_pred))