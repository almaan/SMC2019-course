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

from utils import BSPF

def visualize_loglike(ll,param_vals):
    fig, ax = plt.subplots(1,1,figsize = (9,5))

    xticklabels = ["{:.2f}".format(x) for x in param_vals]

    ax.set_xlabel(r'$\phi$',size = 20)
    ax.set_ylabel('log-likelihood', size = 20)
    bp = ax.boxplot(ll.T, patch_artist = True)

    for patch in bp['boxes']:
        patch.set(facecolor='gray',
                  edgecolor = 'black')

    for patch in bp['medians']:
        patch.set(color = 'darkred')

    for pos in ['top','right']:
        ax.spines[pos].set_visible(False)

    ax.set_xticklabels(xticklabels)

    return fig,ax

def main():
    ddir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'data')
    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    pth = osp.join(ddir,'seOMXlogreturns2012to2014.csv')

    if not osp.exists(odir):
            mkdir(odir)

    with open(pth,'r+') as fopen:
        yvals = fopen.readlines()

    yvals = [float(x.replace('\n','').replace(',','')) for x in yvals]
    yvals = np.array(yvals)

    T  yvals.shape[0]
    niter = 10
    n_particles = 50

    phi_vals = np.arange(0.1,1.1,0.1)

    theta = {'phi':phi_vals,
             'sigma':0.16,
             'beta':0.7,
            }
    x0_dist = st.norm(0,theta['sigma'])
    yt_xt_dist = lambda xt : st.norm(0,(theta['beta']**2)*np.exp(xt))

    log_res = np.zeros((phi_vals.shape[0],niter))

    for k,phi in enumerate(theta['phi']):
        xt_xtm1_dist = lambda xtm1 : st.norm(loc = phi*xtm1,
                                             scale = theta['sigma'])

        bpf = BSPF(xt_xtm1_dist,
                   yt_xt_dist,
                   x0_dist,
                  )
        for it in range(niter):
            res = bpf.run(yvals,n_particles)
            log_res[k,it] = res['loglikelihood']


    fig,ax = visualize_loglike(log_res, phi_vals)
    fig.savefig(osp.join(odir,'ex3-boxplot.png'))


if __name__ == '__main__':
    main()




