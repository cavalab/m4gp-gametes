from glob import glob
import os
import random
import sys

learners = ['M4GP','M4GP_EKF']
problems = ['2w_10a_0.05her','2w_10a_0.1her','2w_10a_0.2her','2w_10a_0.4her',
            '2w_100a_0.05her','2w_100a_0.1her','2w_100a_0.2her','2w_100a_0.4her',
            '2w_1000a_0.05her','2w_1000a_0.1her','2w_1000a_0.2her','2w_1000a_0.4her'
            ]

if len(sys.argv)>1:
    learners = [sys.argv[1]]
    if len(sys.argv)>2:
        problems = ','.split(sys.argv[2])

data_dir = '/home/lacava/data/gametes/'

n_cores = 8
if len(sys.argv)>3:
    n_trials = int(sys.argv[3])
else:
    n_trials = 30

for p in problems:
    print(p)
    # dataset = fetch_data(p,local_cache_dir=data_dir)
# for dataset in glob('/home/lacava/data/regression/*.txt'):
    dataset_name = data_dir + p + '.txt'
    results_path = '/home/lacava/projects/m4gp-gametes-2021/results/'

    for ml in learners:
        print('\t',ml)
       

        for i in range(n_trials):
            save_file = results_path + f'{ml}_{p}_{i}.json'

            job_name = ml + '_' + p + '_' + str(i)
            out_file = results_path + '{JOB_NAME}_%J.out'.format(JOB_NAME=job_name)
            error_file = out_file[:-4] + '.err'
                        #submit job
            bjob_line = ('bsub -o {OUT_FILE} -n {N_CORES} '
                         '-J {JOB_NAME} -R "span[hosts=1]" -q {Q} '
                         '"python {ML}.py {DATASET} {SAVE_FILE} {TRIAL} '
                         '{N_CORES}"'.format(OUT_FILE=out_file,
                                             ERROR_FILE=error_file,
                                             JOB_NAME=job_name,
                                             N_CORES=n_cores,
                                             ML=ml,
                                             DATASET=dataset_name,
                                             SAVE_FILE=save_file,
                                             TRIAL=i,
                                             NCORES=n_cores,
                                             Q='epistasis_long'
                                             ))
            print(bjob_line)
            os.system(bjob_line)
