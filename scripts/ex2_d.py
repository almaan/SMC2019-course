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
from utils import  expNormalize,APF
from ex2_b import load_generated_traj
from ex2_c import make_kf_obj

def make_apf_obj():

    var_xt = 0.5
    var_ytxt = 0.1
    phi = 2.0
    psi = 0.8

    var_yt = var_ytxt + (phi**2)*var_xt

    var_xt_yt = var_xt - (var_xt**2)*(phi**2) / var_yt

    mu_xt_yt = lambda yt,xtm1 : psi*xtm1 + var_xt*phi*(yt - phi*psi*xtm1)/var_yt

    xt_yt_xtm1_dist = lambda yt,xtm1 : st.norm(loc = mu_xt_yt(yt,xtm1),
                                               scale = np.sqrt(var_xt_yt))

    yt_xtm1_dist = lambda xtm1 : st.norm(loc = psi*phi*xtm1,
                                         scale = phi**2*var_xt + var_ytxt)


    yt_xt_dist = lambda xt: st.norm(2*xt,np.sqrt(var_yt))

    x0_dist = st.norm(0,np.sqrt(2))

    apf = APF(xt_yt_xtm1_dist,
              yt_xtm1_dist,
              yt_xt_dist,
              x0_dist)

    return apf


def load_bpf_results():
    indir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')
    mupth = osp.join(indir,'ex2-mu-difference-bpf.tsv')
    varpth = osp.join(indir,'ex2-var-difference-bpf.tsv')
    data_mu = pd.read_csv(mupth, sep = '\t', header = 0, index_col = 0)
    data_var = pd.read_csv(varpth, sep = '\t', header = 0, index_col = 0)
    Ns = np.array([int(x.split('_')[1]) for x in data_mu.columns ])
    ts = data_var.index.values

    return ts, Ns, data_mu.values, data_var.values

def visualize_difference_comp(bpf_mu,
                              bpf_var,
                              apf_mu,
                              apf_var,
                              ns):

    fig, ax = plt.subplots(1,2, figsize = (8,4))
    ax = ax.flatten()
    stats = ['Mean','Variance']

    width = 0.1
    ind = np.arange(ns.shape[0])

    for k,data in enumerate([(bpf_mu,apf_mu),(bpf_var,apf_var)]):
        ax[k].set_xlabel('Particles', size = 8)
        ax[k].set_ylabel('Average Difference', size = 8)
        ax[k].set_xticklabels( [0] + ns.tolist())
        ax[k].set_title(stats[k] + ' Comparision')

        ax[k].bar(ind,data[0],
                  facecolor = 'gray',
                  edgecolor = 'black',
                  width = width,
                  label = 'BPF',
                 )

        ax[k].bar(ind + width,
                  data[1],
                  facecolor = 'white',
                  edgecolor = 'black',
                  width = width,
                  label = 'APF',
                 )

        for pos in ['top','right']:
            ax[k].spines[pos].set_visible(False)

    ax[0].legend()

    fig.tight_layout()

    return fig, ax

def main():
    ts,xs,ys = load_generated_traj()
    KF = make_kf_obj()
    xs_kf, ps_kf = KF.estimate_traj(ys, T = ts[-1] + 1)

    _, Nsm, diff_bpf_mu, diff_bpf_var = load_bpf_results()


    apf = make_apf_obj()

    mu_results = {}
    var_results = {}
    table_results = np.zeros((Nsm.shape[0],2))

    for k,n_particles in enumerate(Nsm):

        res = apf.run(n_particles,ys)

        xs_apf = res['xts']
        aidxs_apf = res['aidxs']

        mu = np.sum(xs_apf*res['weights'],axis = 1)
        var = np.sum(res['weights']*(xs_apf-mu.reshape(-1,1))**2,axis= 1)

        #mu = xs_apf.mean(axis = 1)
        #var = xs_apf.var(axis = 1)

        mu_diff = np.abs(mu - xs_kf)
        var_diff = np.abs(var - ps_kf)

        table_results[k,0] = np.mean(mu_diff)
        table_results[k,1] = np.mean(var_diff)

        mu_results.update({'mu_' + str(n_particles):mu_diff})
        var_results.update({'var_' + str(n_particles):var_diff})

    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    if not osp.exists(odir):
        mkdir(odir)


    comp_fig,  comp_ax = visualize_difference_comp(bpf_mu = diff_bpf_mu.mean(axis=0),
                                                   bpf_var = diff_bpf_var.mean(axis=0),
                                                   apf_mu = table_results[:,0],
                                                   apf_var = table_results[:,1],
                                                   ns = Nsm)

    comp_fig.savefig(osp.join(odir,'ex2-comp-bpf-apf.png'))

    table_results = pd.DataFrame(table_results,
                                 index = Nsm,
                                 columns = ['Mean','Variance'])

    mu_results = pd.DataFrame(mu_results,index = ts)
    var_results = pd.DataFrame(var_results,index = ts)

    mu_results.to_csv(osp.join(odir,
                               'ex2-mu-difference-apf.tsv'),
                      sep = '\t',
                      header = True,
                      index = True)

    var_results.to_csv(osp.join(odir,
                                'ex2-var-difference-apf.tsv'),
                       sep = '\t',
                       header = True,
                       index = True)

    with open(osp.join(odir,'ex2-stat-mean-table-apf.tex'),'w+') as fopen:
        ltx = table_results.to_latex(index = True,
                                     bold_rows = True,
                                     column_format = 'c|c|c')

        ltx = ltx.replace('\\\\','\\\\ \n \\hline')
        fopen.writelines(ltx)

if __name__ == '__main__':
    main()
