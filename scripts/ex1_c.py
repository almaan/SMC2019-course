#!/usr/bin/env python3

import sys
import os.path as osp
from os import mkdir

sys.path.append(osp.abspath(__name__))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from utils import Cauchy, Normal, ImportanceSampler
from ex1 import eval_importance_sampling

def plot_cauchy_vs_normal():

    xx = np.arange(-10,10,0.01)
    cauchy = Cauchy(1)
    normal = st.norm(0,1)

    yy_normal = normal.pdf(xx)
    yy_cauchy = cauchy.pdf(xx)

    fig,ax = plt.subplots(1,1, figsize=(8,5))

    ax.plot(xx,yy_normal,label = r'$\mathcal{N}(0,1)$', color = 'red')
    ax.plot(xx,yy_cauchy, label = r'$Cauchy(\gamma = 1)$',color = 'black')
    ax.fill_between(xx,yy_cauchy,yy_normal, alpha = 0.2, color = 'blue')
    ax.legend()
    for pos in ['top','right']:
        ax.spines[pos].set_visible(False)



    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')
    basename = "cauchy-vs.normal.png"

    if not osp.exists(odir):
        mkdir(odir)

    fig.savefig(osp.join(odir,
                      basename))


def main():

    cauchy = Cauchy(gamma = 1,unscaled = True)
    normal = Normal(0,1)
    niter = 100
    Ns = 3**np.arange(5,12)
    basename = 'boxplot-cauchy-as-target.png'
    eval_importance_sampling(target_dist = cauchy,
                             proposal_dist = normal,
                             Ns = Ns,
                             basename = basename,
                             niter = 100)

    plot_cauchy_vs_normal()



if __name__ == '__main__':
    main()

