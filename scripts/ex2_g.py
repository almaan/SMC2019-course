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
from utils import  expNormalize,APF,to_long_format,visualize_genealogy
from ex2_b import load_generated_traj
from ex2_d import make_apf_obj
from ex2_f import visualize_diveristy, get_unique_primary_ancestors

def visualize_ness_comp(ness_war,N,stepsize = 10):

    fig, ax = plt.subplots(1,1,figsize = (15,4))

    times  = np.arange(0,ness_war.shape[0],stepsize)

    times = np.sort(times)
    tt = times.shape[0]

    ax.plot(times,
            (ness_war/ N)[times],
            's-',
            alpha = 0.3,
            color = 'black',
            markerfacecolor = None,
            markeredgecolor = 'black',
            label = 'With Adaptive \n Resampling',
           )

    ax.set_xticklabels(times[0:tt:10],rotation = 45)
    ax.set_xticks(times[0:tt:10])

    ax.legend()

    return fig, ax

def main(n_particles = 100, T = 250):

    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    if not osp.exists(odir):
        mkdir(odir)

    ts, xs, ys = load_generated_traj()
    apf = make_apf_obj()

    res_wess = apf.run(n_particles,
                       ys,
                       resampler = 'systematic',
                       adaptive_resampling = True)

    xs_apf_wess,_,aidxs_apf_wess,ness_wess = res_wess.values()

    fig_ness,ax_ness = visualize_ness_comp(ness_wess,n_particles)

    fig_ness.savefig(osp.join(odir,'ex2-ness-comp.png'))


    n_uni_wess = get_unique_primary_ancestors(aidxs_apf_wess)

    long_res = to_long_format(xs_apf_wess,
                              aidxs_apf_wess,
                              T = T,
                              )

    fig_gen, ax_gen = visualize_genealogy(**long_res,
                                          window = 5,
                                          xs_true = xs,)

    fig_gen.savefig(osp.join(odir,'ex2-gen-exg.png'))

if __name__ == '__main__':
    main()


