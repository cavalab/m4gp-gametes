from ellyn import ellyn
from sklearn.model_selection import KFold
import pandas as pd
from skrebate import ReliefF
import numpy as np
import copy
import sys

dname = sys.argv[1]
dataset = '../../data/gametes/'+ dname + '.txt'
input_data = pd.read_csv(dataset, sep='\t',)
input_data = pd.read_csv(dataset, sep='\t')
X = input_data.drop('Class', axis=1).values.astype(float)
y = input_data['Class'].values
print('fitting relief...')
#X_filt = ReliefF(n_features_to_select=10, n_neighbors=100).fit_transform(X,y)
X_filt = X
print('running ellyn...')
clf = ellyn(classification=True,class_m4gp=False,class_bool=True,init_trees=False,islands=False,num_islands=10,
            island_gens=50,verbosity=333,prto_arch_on=True,popsize=500,
            selection='lexicase', g=100, max_len=50,max_len_init=10, shuffle_data=True,
            fit_type='F1',ops='+,*,-,/,=,!,<,>,<=,>=,if-then,if-then-else,&,|')

estimators = []
scores=[]

for train,test in KFold(shuffle=True).split(X_filt,y):
    estimators.append(copy.deepcopy(clf))
    estimators[-1].fit(X_filt[train],y[train]) 
    scores.append(estimators[-1].score(X_filt[test],y[test]))

print('n_estimators: ',len(estimators))
best_est = estimators[np.argmax(scores)]

with open('archive_' + dname + '_best.txt','w') as out:
    out.write(best_est.output_archive())
# eff it: print all the archives
for i,est in enumerate(estimators):
    with open('archive_' + dname + str(i) + '.txt','w') as out:
        out.write(est.output_archive())
   
import matplotlib.pyplot as plt

h2 = best_est.plot_archive()
h2.savefig('archive_' + dname + '.pdf')

#plt.show()

