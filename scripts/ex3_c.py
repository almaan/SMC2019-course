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

from utils import InverseGamma, BSPF, expNormalize
from ex3_b import visualize_parameter_dist

from pyro.distributions import InverseGamma as IG


class GBPF:
    def __init__(self,
                     xt_xtm1_dist,
                     yt_xt_dist,
                     x0_dist,
                     resampling_method = 'multinomial',
                    ):

            self.eps = 10e-20

            self.xt_xtm1_dist = xt_xtm1_dist
            self.yt_xt_dist = yt_xt_dist
            self.x0_dist = x0_dist

    def resample(self,ws,n):
        aidx =  np.random.choice(np.arange(n),
                                 p = ws,
                                 size = n,
                                 replace = True)
        return aidx

    def run(self,ys,N,xtc):

        T = ys.shape[0]
        wss = np.zeros((N,T+1))
        xts = np.zeros((N,T+1))

        xts[:,0] = self.x0_dist.rvs(N)
        xts[N-1,:] = xtc[0]
        wss[:,0] = np.ones(N) / N

        for t in range(1,T+1):
            # Resample
            aidx = self.resample(wss[:,t-1],N)
            # Propagate
            xts[:,t]  = self.xt_xtm1_dist(xts[aidx,t-1]).rvs()
            xts[:,0:t] = xts[aidx,0:t]
            xts[N-1,:] = xtc
            # Weight
            wss[:,t] = self.yt_xt_dist(xts[:,t]).logpdf(ys[t-1])
            # Normalize Weights
            wss[:,t] = expNormalize(wss[:,t],inlog = True)

        lineage = np.random.choice(np.arange(N),
                                   p=wss[:,-1],
                                   replace = False)

        selected =  xts[lineage,:]

        return selected


class ParticleGibbsSVM:

    def __init__(self,
                 yts,
                 prior_a = 0.01,
                 prior_b = 0.01,
                 phi = 0.985,
                 n_particles = 200,
                ):

        self.n_params = 2
        self.a = prior_a
        self.b = prior_b
        self.phi = phi
        self.yts = yts
        self.n_particles = n_particles

    def allow_params(self,prm):
        return np.all(prm > 0)

    def run_pf(self,
               prm,
               conditioned = None):

        sigma2, beta2 = prm

        x0_dist = st.norm(0,1)

        yt_xt_dist = lambda xt: st.norm(loc = 0.0,
                                        scale = np.sqrt(beta2*np.exp(xt)))


        xt_xtm1_dist = lambda xtm1 : st.norm(loc = self.phi*xtm1,
                                             scale = np.sqrt(sigma2))

        if conditioned is not None:
            bpf = GBPF(xt_xtm1_dist,
                       yt_xt_dist,
                       x0_dist,
                       )

            traj = bpf.run(self.yts,
                           self.n_particles,
                           conditioned)
        else:
            bpf = BSPF(xt_xtm1_dist,
                       yt_xt_dist,
                       x0_dist,
                       )

            res = bpf.run(self.yts,
                          self.n_particles)

            traj = np.sum(res['xs']*res['weights'], axis = 0)

        return traj

    def marginal(self,xts):

        sup = xts[1::] - self.phi*xts[0:-1]
        sup = np.power(sup,2)
        sup = 0.5*np.sum(sup)


        bup = np.exp(-xts[1::])
        bup = bup*(np.power(self.yts,2))
        bup = 0.5*np.sum(bup)

        T = xts.shape[0] - 1
        a = self.a + 0.5*T

        b_sigma2 = self.b + sup
        b_beta2 = self.b + bup

        ig_sigma2 = IG(a,b_sigma2)
        ig_beta2 = IG(a,b_beta2)

        sigma2 = ig_sigma2.sample().numpy()
        beta2 = ig_beta2.sample().numpy()

        return (sigma2,beta2)

    def run(self, niter,):

        params_list = np.zeros((niter,self.n_params))
        params_list[0,0] = 0.15**2
        params_list[0,1] =  0.6**2
        print(f"Initial params : {params_list[0,:]}")

        xts = np.zeros((niter,self.yts.shape[0]+1))


        xts[0,:] = self.run_pf(params_list[0,:],
                               conditioned = None)

        for i in range(1,niter):

            params_list[i,:] = self.marginal(xts[i-1,:])

            xts[i,:] = self.run_pf(params_list[i,:],
                                   conditioned = xts[i-1,:])

            if i % 10 == 0:
                print(f"Iteration {i} / {niter}")
                print(f"Params : {params_list[i,:]}")

        return {'params_list':params_list}


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

    T = yvals.shape[0]
    n_particles = 200
    niter = 5000
    burn_in = int(0.4*niter)

    np.random.seed(137)

    pgibbs =  ParticleGibbsSVM(yvals,
                               prior_a = 0.01,
                               prior_b = 0.01,
                               phi = 0.985,
                               n_particles = n_particles,
                              )


    res = pgibbs.run(niter)


    sigma_dist = res['params_list'][burn_in::,0]
    beta_dist = res['params_list'][burn_in::,1]

    fig,ax = visualize_parameter_dist(sigma_dist,beta_dist)

    fig.savefig(osp.join(odir,'ex3-hist-params-gibbs.png'))


if __name__ == '__main__':
    main()
