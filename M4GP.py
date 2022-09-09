import multiprocessing
from sklearn.metrics import accuracy_score 
from ellyn import ellyn
import json

m4gp = ellyn(classification=True, 
            class_m4gp=True, 
            islands=True,
            num_islands=8,
            island_gens=20,
            verbosity=2,
            prto_arch_on=True,
            popsize=500,
            g=100,
            selection='lexicase',
            max_len=100,
            rt_cross=0.5,
            rt_mut=0.5,
            scoring_function=accuracy_score,
            fit_type='F1',
            shuffle_data=True)

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    import sys
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from train import train

    # Create the pipeline for the model
    clf = make_pipeline(
            StandardScaler(),
            m4gp
            )

    train(clf,
          'M4GP',
          sys.argv[1],
          sys.argv[2],
          sys.argv[3],
          sys.argv[4],
          )
    
