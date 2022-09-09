# test m4gp_cp
import sys
# sys.path.insert(0,'/home/lacava/code/ellyn/ellyn')
from ellyn import ellyn
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, \
                                    StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from tpot_metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
import itertools
import time
import pdb

dataset = '/home/lacava/data/CGEMS/CGEMS-prostate-cancer-data-only-genes-predict-aggressive.csv'
output_file = '/home/lacava/analysis/gametes/test.csv'
trial = 0
n_cores = 10
# Read the data set into memory
input_data = pd.read_csv(dataset, sep=None, engine='python')
print(input_data.describe())
#header
# with open(output_file,'w') as out:
#     out.write('dataset\tmethod\ttrial\tparameters\taccuracy\ttime\tmodel\n')

# data
sc = StandardScaler()
input_data.rename(columns={'class':'Class'},inplace=True)
X = sc.fit_transform(input_data.drop('Class', axis=1).values.astype(float))
# ytmp = input_data['Class'].values
le = LabelEncoder()
y = le.fit_transform(input_data['Class'].values)

num_islands=10
# Create the pipeline for the model
clf = ellyn(classification=True,class_m4gp=True, islands=True,
            num_islands=num_islands, island_gens=100, verbosity=2,
            prto_arch_on=True,popsize=1000,g=100,selection='lexicase',
            max_len=50,scoring_function=balanced_accuracy_score,
            fit_type='F1')

cv = StratifiedKFold(n_splits=10,shuffle=False)
# 10-fold CV score for the pipeline
t0 = time.time()
clf.fit(X,y)

# scores = cross_val_score(clf,X, y, cv=cv, n_jobs=int(n_cores/num_islands))
mean_time= time.time()-t0
scores = clf.score(X, y)
#fit model

# get fit time


# print results
print('dataset\tmethod\ttrial\tparameters\tbal_accuracy\ttime')
out_text = '\t'.join([dataset.split('/')[-1][:-7],
                      'M4GP',
                      str(trial),
                      str(clf.get_params()),
                      str(np.mean(scores)),
                      str(mean_time)])
# WIP: add printout of pareto archive
# print summary results
with open(output_file,'a') as out:
    out.write(out_text+'\n')
print(out_text)
with open(output_file[:-4]+'.'+str(trial)+'.archive','w') as out:
    out.write(clf.output_archive())
clf.output_archive()

sys.stdout.flush()
