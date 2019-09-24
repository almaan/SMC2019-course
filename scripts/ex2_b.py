#!/usr/bin/env python3

import sys
import os.path as osp
from os import mkdir

sys.path.append(osp.abspath(__name__))

import numpy as np
import scipy.stats as st
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from utils import KalmanFilter

def load_generated_traj():
    indir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'data')
    inpth = osp.join(indir,'ex2-generated-data.tsv')

    if not osp.exists(inpth):
        print(f"The file {inpth} does not exists; make sure to run the script",
              f" associated with the previous task to generate such data")
        sys.exit(-1)

    data = pd.read_csv(inpth,
                       sep = '\t',
                       header = 0,
                       index_col = 0,
                      )

    ys = data['Observed_Values'].values
    xs = data['Hidden_Values'].values
    ts = data['Time'].values

    return ts,xs,ys

def main():

    ts, xs, ys = load_generated_traj()
    niter = 100

    trajs = np.zeros((niter,ts.shape[0]))

    params = dict(A = 0.8,
                  C = 2.0,
                  Q = 0.5,
                  R = 0.1,
                  P0 = 2,
                  x0 = 0)

    KF = KalmanFilter(**params)

    xs_kf,ps = KF.estimate_traj(ys, T = ts[-1] + 1)

    fig,ax = plt.subplots(1,1, figsize = (8,5))

    ax.plot(ts[1::],xs[1::],
            color = 'black',
            linewidth = 6,
            label = 'True')

    ax.plot(ts[1::],
            xs_kf[1::],
            linewidth = 1,
            alpha = 1,
            color = 'red',
            label = 'Kalman Estimate')

    diff = (xs_kf - xs)

    ax.plot(ts[1::],diff[1::],
            color = 'blue',
            linewidth = 1,
            linestyle = 'dashed',
            label = 'Difference')

    ax.legend()

    for pos in ['top','right']:
        ax.spines[pos].set_visible(False)

    ax.set_xlabel('Time',size = 20)
    ax.set_ylabel('Value',size = 20)

    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    if not osp.exists(odir):
        mkdir(odir)


    fig.savefig(osp.join(odir,'ex2-kalman-filter-comp.png'))


if __name__ == '__main__':
    main()





