import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    import sys
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    #### IMPORT RELIEFF
    from skrebate import ReliefF
    from train import train
    from M4GP import m4gp

    # Create the pipeline for the model
    # use RelieifF for filtering
    ekf = ReliefF(n_features_to_select=10, n_neighbors=100)

    clf = make_pipeline(
            StandardScaler(),
            ekf,
            m4gp
            )

    train(clf,
          'M4GP+EKF',
          sys.argv[1],
          sys.argv[2],
          sys.argv[3],
          sys.argv[4],
          )
