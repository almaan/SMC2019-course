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

def hide_spines(ax,sides = None):
    if sides is None:
        sides = ax.spines.keys()

    for pos in sides:
            ax.spines[pos].set_visible(False)


class PiTarget:
    def __init__(self,):
        pass

    def _insite(self,x):
        insite = np.linalg.norm(x,
                                ord = np.inf,
                                axis = 0)

        insite = (insite < 1) & (np.all(x > 0,axis = 0))
        return insite.astype(int)

    def logpdf(self,x):
        p1 = np.log(np.cos(x[0,:]*np.pi)**2)
        p2 = np.log(np.sin(x[1,:]*3*np.pi)**6)
        p3 = -30 * (x[0,:] ** 2 + x[1,:] ** 2)
        p =  p1 + p2 + p3
        p *= self._insite(x)
        p[p == 0] = -np.inf
        return p

    def pdf(self,x):
        return np.exp(self.logpdf(x))

    def temperlogpdf(self,x,k,K):
        return self.logpdf(x) * (k/K)

    def temperpdf(self,x,k,K):
        return np.exp(self.temperlogpdf(x,k,K))

    def sample0(self,N):
        return np.random.uniform(0,1,size = (2,N))


class InvMH:
    def __init__(self,
                 t_dist,
                 sigma = 0.02,
                 ):

        self.td = t_dist
        S = np.eye(2) * sigma**2
        self.pd = lambda xhat : st.multivariate_normal(xhat,S)

    def run(self,
            xk,
            niter,
            k,
            K,
           ):

        zs = xk.reshape(-1,1)
        zsmp = self.td.temperlogpdf(zs,k,K)

        for i in range(0,niter):
            _zs = self.pd(zs.reshape(-1,)).rvs().reshape(-1,1)
            _zsmp = self.td.temperlogpdf(_zs,k,K)

            alpha = np.exp(_zsmp - zsmp)
            alpha[np.isnan(alpha)] = 0
            alpha = np.min((1,alpha))

            u = np.random.random()

            if u <= alpha:
                zs = _zs[:]
                zsmp = _zsmp

        return zs.reshape(-1,)

def main():

    # Set seed for reproducibilty
    np.random.seed(1337)

    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    # Analysis parameters
    K = 100
    n_particles = 100
    sigma = 0.02
    ess_thrs = 0.7*n_particles

    # Variable Holders
    essv = np.zeros(K)
    xks = np.zeros((K,2,n_particles))
    wss = np.zeros((K,n_particles))
    logzratio = np.zeros(K)
    rs_t = 0

    # Define distributions and sampler
    gamma = PiTarget()
    sampler = InvMH(t_dist = gamma,
                    sigma = sigma)

    # Set ess definition
    get_ess = lambda w : 1.0 / (w**2).sum()

    # Initialize
    xks[0,:,:] = gamma.sample0(n_particles)
    wss[0,:] = np.ones(n_particles) / n_particles
    essv[0] = get_ess(wss[0,:])

    # Tempering and sequential sampling
    for k in range(1,K):
        if k % 10 == 0 and k >= 10:
            print(f"k : {k} / {K}")
            print(f'ESS is ' + str(essv[k-1]))


        zr = gamma.temperlogpdf(xks[k-1,:,:],k,K)
        zr -= gamma.temperlogpdf(xks[k-1,:,:],k-1,K)
        zr += np.log(wss[k-1,:])
        wss[k,:] = expNormalize(zr,inlog = True)

        essv[k] = get_ess(wss[k,:])

        if essv[k] < ess_thrs:
            print(f"Time since last resampling : {k - rs_t}")
            rs_t = k
            aidx = np.random.choice(np.arange(n_particles),
                                    p = wss[k,:],
                                    replace = True,
                                    size = n_particles,
                                    )

            xks[k,:,:] = xks[k-1,:,aidx].reshape((1,2,n_particles))
            wss[k,:] = 1/n_particles
        else:
            xks[k,:,:] = xks[k-1,:,:]

        for p in range(n_particles):
            xks[k,:,p] = sampler.run(xks[k,:,p],1,k,K)

        logzratio[k] = np.log(np.exp(zr).sum())

        zratio = np.exp(logzratio.sum())

    print("FINAL")
    print(zratio)
    true_val = 0.00648817
    diff = np.abs(zratio - true_val) / np.abs(true_val)
    print(f"Difference ratio True : {diff}")

    # Visualize and save results --------

    # Save Estimated Z-value
    with open(osp.join(odir,'ex4-z-value.txt'),'w+') as fopen:
        fopen.write(str(zratio.round(4)))

    # Particle distribution
    n_show = 4
    select = np.linspace(0,K-1,n_show).astype(int)

    fig, ax = plt.subplots(2,int(np.ceil((n_show) / 2)))
    ax = ax.flatten()

    xx, yy = np.meshgrid(np.arange(0,1,0.01),
                         np.arange(0,1,0.01))

    pi_pdf = lambda x,y : np.cos(np.pi * x)**2 * \
                          np.sin(y*3*np.pi) * \
                          np.exp(-30*(x**2 + y**2))

    zz = pi_pdf(xx,yy)

    for pp in range(n_show):

        ax[pp].set_title(select[pp])
        ax[pp].contourf(xx,yy,zz, cmap = plt.cm.inferno)
        ax[pp].scatter(x = xks[select[pp],0,:],
                       y = xks[select[pp],1,:],
                       c = 'purple',
                       s = 20,
                       edgecolor = 'white',
                       alpha = 0.7,
                      )

        ax[pp].set_xlim([0,1])
        ax[pp].set_ylim([0,1])
        ax[pp].set_xticks([0,1])
        ax[pp].set_yticks([0,1])

        hide_spines(ax[pp])

    fig.tight_layout()
    fig.savefig(osp.join(odir,'ex4-particle-position.png'))

    # ESS as a function of k

    fig, ax = plt.subplots(1,1,figsize = (7,4))

    ax.plot(np.arange(K),
            essv,
            'o-',
            color = 'black',
            markerfacecolor = 'gray',
            markeredgecolor = 'black',
           )

    ax.set_ylabel('ESS', fontsize = 16)
    ax.set_xlabel('k',fontsize = 15)

    hide_spines(ax,['right','top'])

    fig.savefig(osp.join(odir,'ex4-ness-vs-k.png'))


if __name__ == '__main__':
    main()

