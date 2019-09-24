#!/usr/bin/env python3

import sys
import os.path as osp
from os import mkdir

sys.path.append(osp.abspath(__name__))

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

from utils import Cauchy, Normal, ImportanceSampler

def eval_importance_sampling(target_dist,
                             proposal_dist,
                             Ns,
                             basename,
                             niter = 100,
                             ):

    np.random.seed(1337)
    IS = ImportanceSampler(target_dist, proposal_dist)
    Zhatmat = np.zeros((Ns.shape[0],niter))

    for k,n in enumerate(Ns):
        for it in range(niter):

            X = IS.sample(n)
            W = IS.compute_weights(X)

            Zhat = np.mean(W)
            Zhatmat[k,it] = Zhat

    Ztrue = np.sqrt(np.pi * 2)
    diff = Ztrue - Zhatmat

    fig, ax = plt.subplots(1,1, figsize = (8,5))
    sns.boxplot(data = diff.T,
                ax = ax,
                palette = sns.color_palette("Spectral",Ns.shape[0]),
                )

    for pos in ['top','right']:
        ax.spines[pos].set_visible(False)

    ax.set_xticklabels(Ns,size = 12)
    ax.set_yticklabels(ax.get_yticks().round(2),size = 12)
    ax.set_xlabel('Particles (N)',size = 17)
    ax.set_ylabel(r'$\hat{Z} - Z$',size = 17)

    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    if not osp.exists(odir):
        mkdir(odir)

    fig.savefig(osp.join(odir,
                      basename))


def main():
    Ns = 3**np.arange(5,12)
    basename = 'boxplot-normalizing-constant.png'

    cauchy = Cauchy(gamma = 1/np.sqrt(2))
    normal = Normal(0,1,unscaled = True)

    eval_importance_sampling(target_dist = normal,
                             proposal_dist = cauchy,
                             Ns = Ns,
                             basename = basename,
                             niter = 100)



if __name__ == '__main__':
    main()
