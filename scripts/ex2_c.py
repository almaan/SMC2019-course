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
from utils import KalmanFilter,ImportanceSampler, expNormalize, BSPF
from ex2_b import load_generated_traj

def make_kf_obj():
    params = dict(A = 0.8,
                  C = 2.0,
                  Q = 0.5,
                  R = 0.1,
                  P0 = 2,
                  x0 = 0)

    KF = KalmanFilter(**params)
    return KF


def main():
    ts,xs,ys = load_generated_traj()

    xt_xtm1 = lambda xtm1 : st.norm(0.8*xtm1,np.sqrt(0.5))
    yt_xt = lambda xt: st.norm(2*xt,np.sqrt(0.1))
    x0 = st.norm(0,np.sqrt(2))

    KF = make_kf_obj()

    xs_kf,ps_kf = KF.estimate_traj(ys, T = ts[-1] + 1)

    bspf = BSPF(xt_xtm1_dist = xt_xtm1,
                yt_xt_dist = yt_xt,
                x0_dist = x0,
               )

    ns = np.array([10,50,100,2000,5000]).astype(int)
    mu_results = {}
    var_results = {}
    table_results = np.zeros((ns.shape[0],2))

    width = ns.shape[0]*3 + 0.4
    height = 2*3 + 0.4

    fig,ax = plt.subplots(2,ns.shape[0], sharey = 'row',figsize = (width,height))

    for aa in ax.flatten():
        aa.yaxis.set_tick_params(labelleft=True)

    for k,n in enumerate(ns):
        res = bspf.run(ys,n)

        mu = np.sum(res['xs']*res['weights'],axis = 0)
        var = np.sum(res['weights']*(res['xs']-mu)**2,axis= 0)

        mu_diff = np.abs(mu - xs_kf)
        var_diff = np.abs(var - ps_kf)

        table_results[k,0] = np.mean(mu_diff)
        table_results[k,1] = np.mean(var_diff)

        mu_results.update({'mu_' + str(n):mu_diff})
        var_results.update({'var_' + str(n):var_diff})


        ax[0,k].plot(ts[1::],mu_diff[1::],color = 'black')
        ax[1,k].plot(ts[1::],var_diff[1::],color = 'black')

        ax[0,k].set_ylabel(r'$|\mu_{KF}-\mu_{PF}|$')
        ax[1,k].set_ylabel(r'$|\mathrm{Var}_{KF}-\mathrm{Var}_{PF}|$')
        ax[0,k].set_xlabel('Time')
        ax[1,k].set_xlabel('Time')
        ax[0,k].set_title(f"{n} Particles")

        for pos in ['top','right']:
            for r in range(ax.shape[0]):
                ax[r,k].spines[pos].set_visible(False)


    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    if not osp.exists(odir):
        mkdir(odir)

    fig.savefig(osp.join(odir,'ex2-kalman-vs-pf.png'))

    mu_results = pd.DataFrame(mu_results,index = ts)
    var_results = pd.DataFrame(var_results,index = ts)
    table_results = pd.DataFrame(table_results,
                                 index = ns,
                                 columns = ['Mean','Variance'])

    mu_results.to_csv(osp.join(odir,'ex2-mu-difference-bpf.tsv'),sep = '\t', header = True, index = True)
    var_results.to_csv(osp.join(odir,'ex2-var-difference-bpf.tsv'),sep = '\t', header = True, index = True)

    with open(osp.join(odir,'ex2-stat-mean-table.tex'),'w+') as fopen:
        ltx = table_results.to_latex(index = True,bold_rows = True,column_format = 'c|c|c')
        ltx = ltx.replace('\\\\','\\\\ \n \\hline')
        fopen.writelines(ltx)

if __name__ == '__main__':
    main()
