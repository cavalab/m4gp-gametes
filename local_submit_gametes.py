from glob import glob
import os
import random
from pmlb import fetch_data, dataset_names
import sys

learners = ['KerasNN']
problems = ['2w_10a_0.05her','2w_10a_0.1her','2w_10a_0.2her','2w_10a_0.4her',
            '2w_100a_0.05her','2w_100a_0.1her','2w_100a_0.2her','2w_100a_0.4her',
            '2w_1000a_0.05her','2w_1000a_0.1her','2w_1000a_0.2her','2w_1000a_0.4her',
            '2w_5000a_0.05her','2w_5000a_0.1her','2w_5000a_0.2her','2w_5000a_0.4her',]

if len(sys.argv)>1:
    learners = [sys.argv[1]]
    if len(sys.argv)>2:
        problems = []
        problems.append(sys.argv[2])



data_dir = '/media/bill/Drive/Dropbox/PostDoc/data/gametes/'

n_cores = 10
if len(sys.argv)>3:
    n_trials = int(sys.argv[3])
else:
    n_trials = 30

for p in problems:
    print(p)
    # dataset = fetch_data(p,local_cache_dir=data_dir)
# for dataset in glob('/home/lacava/data/regression/*.txt'):
    dataset_name = data_dir + p + '.txt'
    results_path = '/media/bill/Drive/Dropbox/PostDoc/results/gametes/'

    for ml in learners:
        print('\t',ml)
        save_file = results_path + ml + '_' + p + '.csv'
        #write header
        with open(save_file,'w') as out:
            out.write('dataset\tmethod\ttrial\tparameters\tbal_accuracy\ttime\n')
        for i in range(n_trials):

            job_name = ml + '_' + p + '_' + str(i)
             #submit job
            bjob_line = ('python {ML}.py {DATASET} {SAVE_FILE} {TRIAL} '
                         '{N_CORES}'.format(N_CORES=n_cores,
                                            ML=ml,
                                            DATASET=dataset_name,
                                            SAVE_FILE=save_file,
                                            TRIAL=i,))
            print(bjob_line)
            os.system(bjob_line)
