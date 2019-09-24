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
from utils import GaussianStateSpaceModel


def generate_synthetic_data():

    init = st.norm(0,np.sqrt(2))
    Vt = st.norm(0,np.sqrt(0.5))
    Et = st.norm(0,np.sqrt(0.1))
    a = 0.8
    b = 2.0
    T = 2000

    ss = GaussianStateSpaceModel(a,b, Vt = Vt, Et = Et, X0 = init)
    ts,xs,ys = ss.generate_trajectory(T)

    data = dict(Time = ts,
                Hidden_Values = xs,
                Observed_Values = ys,
               )

    data = pd.DataFrame(data)


    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'data')
    if not osp.exists(odir):
        mkdir(odir)

    data.to_csv(osp.join(odir,'ex2-generated-data.tsv'),
                sep = '\t',
                header = True,
                index = True,
                )

    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(ts,ys, color = 'black', linewidth = 3, alpha = 0.7, label = 'Observed')
    ax.plot(ts,xs, color = 'red', linestyle = 'dashed', label = 'Hidden')

    for pos in ['top','right']:
        ax.spines[pos].set_visible(False)

    ax.set_ylabel('Value')
    ax.set_xlabel('Time')
    ax.legend()

    fig.savefig(osp.join(odir,'ex2-generated-data.png'))

if __name__ == '__main__':
    generate_synthetic_data()







