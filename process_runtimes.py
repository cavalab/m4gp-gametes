import pandas as pd
import numpy as np
from glob import glob 

methods = ['ekf_mdr','mdr_only','xgb','logit']

atts = ['a10','a100','a1000','a5000']

her = ['h005','h01','h02','h04']

nice_m = {'ekf_mdr':'TPOT-MDR+EKF','mdr_only':'TPOT-MDR','xgb':'XGBoost','logit':'LR'}

nice_d = {'a10':'10a','a100':'100a','a1000':'1000a','a5000':'5000a',
        'h005':'0.05', 'h01':'0.1', 'h02':'0.2', 'h04':'0.4'}

wf = 'tpot_runtimes.tsv'
dp = '../../results/gametes/tpot'

with open(wf,'w') as out:
    out.write('\t'.join(['method','dataset','time'+'\n']))

for m in methods:
    for a in atts:
        for h in her:
            fnames = glob('/'.join([dp,m,a,h,'*.out']))
            if len(fnames)==0:
                fnames = glob('/'.join([dp,m,a,h,'*.log']))
            print(fnames)
            for f in fnames:
                time = 0
                with open(f,'r') as cf:

                    for line in cf:
                        
                        if 'mdr' in m:
                            if 'Time lapsed:' in line:
                                time = float(line[12:])
                        else:
                            if 'CPU time :' in line:
                                time = float(line[16:-5])

                        if time != 0:
                            output = '\t'.join([nice_m[m],
                                               '_'.join(['2w',nice_d[a],nice_d[h]]),
                                                str(time)+'\n'])
                            print(output)
                            with open(wf,'a') as out:                       
                                out.write(output)
                            time = 0
