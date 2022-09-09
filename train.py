import json
import sys
# from ellyn import ellyn
from M4GP import m4gp
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
### IMPORT RELIEFF
from skrebate import ReliefF
import itertools
import time

def train(clf, clf_name, dataset, output_file, trial, n_cores):

    # Read the data set into memory
    input_data = pd.read_csv(dataset, sep=None, engine='python')
    X = input_data.drop('Class', axis=1).values.astype(float)
    # ytmp = input_data['class'].values
    le = LabelEncoder()
    y = le.fit_transform(input_data['Class'].values)

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.3, 
                                                        train_size=0.7,
                                                        shuffle=True
                                                        )

    # train/test split score for the pipeline
    t0 = time.time()
    clf.fit(X_train, y_train)
    score_train = accuracy_score(y_train, clf.predict(X_train)) 
    score_test = accuracy_score(y_test, clf.predict(X_test)) 
    mean_time= time.time()-t0
    # print results
    df = {
        'dataset':dataset.split('/')[-1][:-7],
        'method': clf_name,
        'trial': trial,
        'accuracy_train': score_train,
        'accuracy_test': score_test,
        'time':mean_time
        }
    print(df)
    with open(output_file, 'w') as out:
        json.dump(df, out)

    sys.stdout.flush()
