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

def visualize_diveristy(n_uni_sys,
                        n_uni_mult,
                       labels):

    fig, ax = plt.subplots(1,1,
                           figsize = (8,4))

    ax.plot(np.arange(n_uni_sys.shape[0]),
            n_uni_sys,
            'o-',
            markersize = 1,
            color = 'gray',
            label = labels[0])

    ax.plot(np.arange(n_uni_sys.shape[0]),
            n_uni_mult,
            'o-',
            markersize = 1,
            color = 'black',
            label = labels[1])


    for pos in ['top','right']:
        ax.spines[pos].set_visible(False)

    ax.legend()
    ax.set_ylabel('Unique Primary \n'
                  'ancestor indices')

    ax.set_xlabel('Time')

    return fig, ax

def get_unique_primary_ancestors(aidxs):

    tree = np.zeros(aidxs.shape)

    tree[0,:] = aidxs[0,:]

    n_uni = []

    for t in range(1,aidxs.shape[0]):
        tree[t,:] = tree[t-1,aidxs[t,:]]
        n_uni.append(np.unique(tree[t,:]).shape[0])

    n_uni = np.array(n_uni)

    return n_uni

def main(n_particles = 100, T = 250):

    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    if not osp.exists(odir):
        mkdir(odir)

    ts, xs, ys = load_generated_traj()
    apf = make_apf_obj()

    res_sys = apf.run(n_particles,
                                         ys,
                                         resampler = 'systematic')


    xs_apf_sys = res_sys['xts']
    aidxs_apf_sys = res_sys['aidxs']

    res_mult  = apf.run(n_particles,
                        ys,
                        resampler = 'multinomial')

    xs_apf_mult = res_mult['xts']
    aidxs_apf_mult= res_mult['aidxs']

    n_uni_sys = get_unique_primary_ancestors(aidxs_apf_sys)
    n_uni_mult = get_unique_primary_ancestors(aidxs_apf_mult)


    fig_uni, ax_uni = visualize_diveristy(n_uni_sys,
                                          n_uni_mult,
                                          labels = ['systematic','multinomial'])


    fig_uni.savefig(osp.join(odir,'ex2-sys-vs-mult.png'))

    long_res = to_long_format(xs_apf_sys,
                              aidxs_apf_sys,
                              T = T,
                              )

    fig_gen, ax_gen = visualize_genealogy(**long_res,
                                  window = 5)

    fig_gen.savefig(osp.join(odir,'ex2-gen-exf.png'))

if __name__ == '__main__':
    main()



