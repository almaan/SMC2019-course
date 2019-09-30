#!/usr/bin/env python3

import sys
import os.path as osp
from os import mkdir

sys.path.append(osp.abspath(__name__))

import numpy as np
import scipy.stats as st
from scipy.special import loggamma
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from utils import BSPF, InverseGamma

def visualize_parameter_dist(sigma_dist, beta_dist):

    fig, ax = plt.subplots(1,2,figsize = (8,4))

    ax[0].hist(sigma_dist,
               edgecolor = 'black',
               facecolor = 'gray',
               alpha = 0.3,
               density = True,
              )

    ax[0].axvline(x = np.mean(sigma_dist),
                  color = 'darkred',
                  alpha = 0.3,
                  )

    ax[0].set_title(r'$\sigma^2$')

    ax[1].hist(beta_dist,
               edgecolor = 'black',
               facecolor = 'gray',
               alpha = 0.3,
               density = True,
              )

    ax[1].axvline(x =np.mean(beta_dist),
                 color = 'darkred',
                 alpha = 0.3,
                 )

    ax[1].set_title(r'$\beta^2$')

    return fig,ax


def eval_dist(data,
              params,
              phi = 0.985,
              n_particles = 200):

    sigma2 = params[0]
    beta2 = params[1]

    def x0_dist():
        return st.norm(0,np.sqrt(sigma2))

    def yt_xt_dist(xt):
        return st.norm(0, np.sqrt(beta2*np.exp(xt)))

    def xt_xtm1_dist(xtm1):
        return st.norm(phi*xtm1,np.sqrt(sigma2))

    bpf = BSPF(xt_xtm1_dist,
               yt_xt_dist,
               x0_dist,
               )

    res = bpf.run(data,n_particles)

    return res['loglikelihood']



class PMH:
    def __init__(self,
                 data,
                 n_particles):

        self.ys = data
        self.S = np.eye(2)*(0.01**2)
        self.prior = InverseGamma(0.01,0.01)
        self.n_particles = n_particles

    def run(self,niter):

        prms = np.zeros((niter,2))
        prms[0,0] = np.random.uniform(0,0.3)
        prms[0,1] = np.random.uniform(0,1)

        print(f"Initial Parameters | {prms[0,:]}")


        pps = np.zeros(niter)
        pps[0] = eval_dist(self.ys,
                       prms[0,:],
                       n_particles = self.n_particles)

        priors = np.zeros(niter)
        priors[0] = self.prior.logpdf(prms[0,:]).sum()

        for ii in range(1,niter):
            prms[ii,:] = prms[ii-1,:]
            prms[ii,:] += np.random.multivariate_normal(np.zeros(2),
                                                        cov = self.S)

            if np.all(prms[ii,:] > 0):
                pps[ii] = eval_dist(self.ys,
                                    prms[ii,:],
                                    n_particles = self.n_particles)

                priors[ii] = self.prior.logpdf(prms[ii,:]).sum()

                alpha = np.min((1,np.exp(pps[ii] + \
                                         priors[ii] - \
                                         pps[ii-1] - \
                                         priors[ii-1])))

                u = np.random.random()
            else:
                alpha = 0
                u = np.inf

            if u > alpha:
                 prms[ii,:] = prms[ii-1,:]
                 priors[ii] = priors[ii-1]
                 pps[ii] = pps[ii-1]
            else:
                 print(f"Iteration {ii} | Params {prms[ii,:]}")


        return {'params_list':prms}

def main():

    ddir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'data')
    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    pth = osp.join(ddir,'seOMXlogreturns2012to2014.csv')

    if not osp.exists(odir):
            mkdir(odir)

    with open(pth,'r+') as fopen:
        yvals = fopen.readlines()

    yvals = [float(x.replace('\n','').replace(',','')) for \
             x in yvals]

    yvals = np.array(yvals)

    T = yvals.shape[0]


    niter= 2500
    burn_in = int(0.4*niter)
    n_particles = 1000


    np.random.seed(1337)

    pmh = PMH(yvals,
              n_particles)

    res = pmh.run(niter)

    sigma_dist = res['params_list'][burn_in::,0]
    beta_dist = res['params_list'][burn_in::,1]

    filenames = ['ex3-'+ x + '-mean.txt' for x in ['sigma2','beta2']]
    meanvals = [x.mean() for x in [sigma_dist, beta_dist]]

    for var in range(2):
        with open(osp.join(odir,filenames[var]),'w+') as fopen:
            fopen.write(str(meanvals[var]))

    fig,ax = visualize_parameter_dist(sigma_dist,beta_dist)

    fig.savefig(osp.join(odir,'ex3-hist-params-mh.png'))

if __name__ == '__main__':
    main()
