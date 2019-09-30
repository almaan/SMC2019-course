#!/usr/bin/env python3

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.special import loggamma



class InverseGamma:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.log_z = a*np.log(self.b) - loggamma(self.a)

    def rvs(self,):
        pass

    def logpdf(self,x):
        return self.log_z + np.log(x) * (-self.a - 1) - self.b/x

    def pdf(self,x):
        return np.exp(self.logpdf(x))


class BSPF:
    def __init__(self,
                 xt_xtm1_dist,
                 yt_xt_dist,
                 x0_dist,
                 resampling_method = 'multinomial',
                ):

        self.eps = 10e20

        self.xt_xtm1_dist = xt_xtm1_dist
        self.yt_xt_dist = yt_xt_dist

        self.x0_dist = x0_dist

        self.resample = eval(''.join(['self.',
                                     resampling_method,
                                     '_resampling']
                                    ))

    def multinomial_resampling(self,ws,n):

        if np.abs(ws.sum() - 1) > self.eps:
            print('Normalizing Weights')
            ws = expNormalize(ws,inlog = True)

        nxs =  np.random.choice(np.arange(n),
                                p = ws,
                                size = n,
                                replace = True)
        return nxs.astype(int)

    def run(self,ys,N):

        ll = 0.0
        T = ys.shape[0]
        wss = np.zeros((N,T+1))
        xts = np.zeros((N,T+1))
        aidxs = np.zeros((N,T+1)).astype(int)

        xt = self.x0_dist().rvs(N)
        xts[:,0] = xt
        wss[:,0] = np.ones(N) / N

        for t in range(1,T+1):
            # Resample
            aidx = self.resample(wss[:,t-1],N)
            aidxs[:,1:(t-1)] = aidxs[aidx,1:(t-1)]
            aidxs[:,t] = aidx
            # Propagate
            xts[:,t] = self.xt_xtm1_dist(xts[aidx,t-1]).rvs()
            # Weight
            wss[:,t] = self.yt_xt_dist(xts[:,t]).logpdf(ys[t-1])
            # Compute LL
            ll += np.log(np.exp(wss[:,t]).sum()) - np.log(N)
            # Normalize Weights
            wss[:,t] = expNormalize(wss[:,t],inlog = True)

        res = {'xs':xts,
               'weights':wss,
               'loglikelihood':ll,
               'aidxs':aidxs.astype(int),
              }

        return res


def to_long_format(xs,
                   aidx,
                   T = None):

    if T is None:
        T = xs.shape[0]

    N = xs.shape[1]

    survivors = aidx[0,:]
    for t in range(1,xs.shape[0]):
        survivors = survivors[aidx[t,:]]

    edges = []
    sedges = []
    coords = []
    origin = aidx[0,:]

    for t in range(T):
        origin = origin[aidx[t,:]]
        opos = len(coords)
        for k,n in enumerate(aidx[t,:]):
            coords.append((t,xs[t,k]))
            if len(coords) > N:
                source = opos - N  + int(n)
                target = opos + k
                if origin[k] in survivors and k in aidx[t+1,:]:
                    sedges.append((source,target))
                else:
                    edges.append((source,target))

    edges = np.array(edges).astype(int)
    sedges = np.array(sedges).astype(int)
    coords =  np.array(coords)

    long_pos = coords[:,1].flatten()
    long_time = coords[:,0].flatten()

    return {'xst':long_pos,
            'time':long_time,
            'dead_edges':edges,
            'alive_edges':sedges}


def visualize_genealogy(xst,
                        time,
                        dead_edges,
                        alive_edges,
                        window,
                        xs_true = None,
                        ):

    fig, ax = plt.subplots(1,3, figsize = (15,6))
    tmax = time.max()
    delta = 0.5

    tlims = [[-delta*5 ,tmax + delta*5] ,
             [-delta,window + delta],
             [tmax-window-delta,tmax + delta]
            ]

    titles = ['Full','Initial','End']

    for k in range(3):

        ax[k].set_xlim(tlims[k])
        ax[k].set_title(titles[k])

        ax[k].plot(time[dead_edges].T,
                 xst[dead_edges].T,
                 '-',
                 color = 'black',
                 alpha = 0.2,
                )

        ax[k].plot(time[alive_edges].T,
                 xst[alive_edges].T,
                 '-',
                 color = 'red',
                 alpha =0.2,
                )

        nitems = dead_edges.shape[0] + alive_edges.shape[0]
        order = np.arange(nitems)
        np.random.shuffle(order)

        for l,p in zip(ax[k].get_lines(),order):
            l.set_zorder(p)

        ax[k].plot(time,
                   xst,
                   'o',
                   markerfacecolor = None,
                   markeredgecolor = 'black',
                   alpha = 0.1,
                   markersize = 2,
                  )

        for pos in ['top','right']:
            ax[k].spines[pos].set_visible(False)

        ax[k].set_xlabel('Time')
        ax[k].set_ylabel(r'State $x_t$')

        tmin = time.min().astype(int)
        tmax = time.max().astype(int)

        if xs_true is not None:
            ax[k].plot(np.arange(tmin,tmax+1),
                       xs_true[tmin:tmax+1],
                       '--',
                       color = 'blue',
                       alpha = 0.2,
                       label = 'True',
                       )

    fig.tight_layout()

    return fig, ax



class APF:
    def __init__(self,
                 xt_yt_xtm1_dist,
                 yt_xtm1_dist,
                 yt_xt_dist,
                 x0_dist,
                ):

        self.yt_xtm1_dist = yt_xtm1_dist
        self.xt_yt_xtm1_dist = xt_yt_xtm1_dist
        self.yt_xt_dist = yt_xt_dist
        self.x0 = x0_dist

    def _omega(self,xt,yt,xtm1):
        return np.ones(xt.shape[0])/xt.shape[0]

    def _nu_yt_xtm1(self,yt,xtm1,islog =False):
        nu_tm1 = self.yt_xtm1_dist(xtm1).logpdf(yt)

        if not islog:
            nu_tm1 = np.exp(nu_tm1)

        return nu_tm1

    def systematic(self,ww):

         N = len(ww)
         cs = np.cumsum(ww)
         a = np.zeros(N, 'i')
         u  = (np.random.random() + np.arange(N)) / N
         r, s = 0, 0

         while r < N:
             if u[r] < cs[s]:
                 a[r] = s
                 r += 1
             else:
                 s += 1

         return a


    def multinomial(self,ww):
        cs = np.cumsum(ww)
        cs[-1] = 1.
        u =  np.random.random(len(ww))
        a = np.searchsorted(cs,u)

        return a


    def _set_resampler(self,name):
        allowed = ['multinomial',
                   'stratified',
                   'systematic',
                  ]

        if name in allowed:
            self.resample = eval('self.' + name)
            print('Using {} resampling'.format(name))

        elif name is None:
            self.resample = lambda w: np.arange(w.shape[0])
        else:
            self.resample = self.multinomial
            print('Using multinomial resampling')


    def run(self,N,ys,
            resampler = 'multinomial',
            adaptive_resampling = False,
            ess_thrs = 50):

        self._set_resampler(resampler)

        if not adaptive_resampling:
            ess_thrs = np.inf

        T = ys.shape[0]
        xtmat = np.zeros((T,N))
        wss = np.zeros((T,N))
        wss[0,:] = np.ones(N) / N

        xt = self.x0.rvs(N)
        xtmat[0,:] = xt

        aidxs = np.zeros((T,N),dtype = np.int)
        aidxs[0,:] = np.arange(N).astype(int)

        ess_vec = np.zeros(T)
        ess = N
        ess_vec[0] = ess

        for t in range(1,T):

            # Resample
            if ess < ess_thrs:
            # Compute adjustment mulitpliers
                nus = self._nu_yt_xtm1(ys[t],xtmat[t-1,:],islog = True)
                nus = expNormalize(nus + np.log(wss[t-1,:]),inlog = True)
                aidx = self.resample(nus)
                wss[t-1,:] = np.ones(N) / N
            else:
                aidx= np.arange(N)

            aidxs[t,:] = aidx

            # Propagate
            xtmat[t,:] = self.xt_yt_xtm1_dist(ys[t],xtmat[t-1,aidx]).rvs()

            # Weight
            if not adaptive_resampling:
                wss[t,:] = 1/N
            else:
                wss[t,:] = self.xt_yt_xtm1_dist(ys[t],xtmat[t-1,aidx]).logpdf(xtmat[t,:])
                wss[t,:] = expNormalize(wss[t,:] + np.log(wss[t-1,:]),inlog = True)

            # Compute effective sample size
            ess = 1.0 / (wss[t,:] ** 2).sum()
            ess_vec[t] = ess

        return {'xts':xtmat,
                'weights':wss,
                'aidxs':aidxs,
                'ess':ess_vec,
               }

def expNormalize(ws, inlog = False):
    if not inlog:
        ws = np.log(ws)

    wmax = ws.max()
    ws = np.exp(ws - wmax)
    ws = ws / ws.sum()

    return ws

class KalmanFilter:
    def __init__(self,
                 A,
                 C,
                 Q,
                 R,
                 P0,
                 x0,
                ):

        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.P0 = P0
        self.x0 = x0

    def _getKt(self,Pt_tm1,):
       Kt = Pt_tm1*self.C / (self.C*Pt_tm1*self.C + self.R)
       return Kt
    def _getPt_tm1(self,Ptm1_tm1):
       Pt_tm1 = self.A*Ptm1_tm1*self.A + self.Q
       return Pt_tm1

    def _getPt_t(self,Pt_tm1,Kt):
        Pt_t = Pt_tm1 - Kt*self.C*Pt_tm1
        return Pt_t
    def _getxt_t(self,xtm1_tm1,yt,Kt):
        xt_t = self.A*xtm1_tm1 + Kt*(yt-self.C*self.A*xtm1_tm1)
        return xt_t

    def estimate_traj(self,ys,T = None):

        if not T:
            T = ys.shape[0]

        xs,ps = [],[]
        Pt_t = self.P0
        xt_t = self.x0
        xs.append(xt_t)
        ps.append(Pt_t)

        for t in range(1,T):
            Pt_tm1 = self._getPt_tm1(Pt_t)
            Kt = self._getKt(Pt_tm1)

            Pt_t = self._getPt_t(Pt_tm1,Kt)
            xt_t = self._getxt_t(xt_t,ys[t],Kt)
            xs.append(xt_t)
            ps.append(Pt_t)

        xs = np.array(xs)
        ps = np.array(ps)
        return xs,ps



class GaussianStateSpaceModel:
    def __init__(self,
                 xtm1_coef,
                 yt_coef,
                 Vt,
                 Et,
                 X0,
                ):

        self.a = xtm1_coef
        self.b = yt_coef

        self.Vt = Vt
        self.Et = Et
        self.X0 = X0

    def xt_xtm1(self,xtm1):
        return self.a*xtm1 + self.Vt.rvs()

    def yt_xt(self,xt):
        return self.b*xt + self.Et.rvs()

    def generate_trajectory(self,T):

        ts,xs,ys = [],[],[]

        xt = self.X0.rvs()
        yt = self.yt_xt(xt)

        xs.append(xt)
        ys.append(yt)
        ts.append(0)

        for t in range(1,T):
            xt = self.xt_xtm1(xt)
            yt = self.yt_xt(xt)

            xs.append(xt)
            ys.append(yt)
            ts.append(t)

        for it in [xs,ys,ts]:
            it = np.array(it)

        return ts,xs,ys



class Cauchy:
    def __init__(self,
                 gamma,
                 unscaled = False,
                ):

        self.gamma = gamma

        if unscaled:
            self.Z = 1.0
        else:
            self.Z = np.pi

    def rvs(self,N):
        u = np.random.uniform(0,1,size = N)
        x = self.gamma * np.tan(np.pi*(u-0.5))
        return x

    def logpdf(self,x):
        return np.log(self.gamma) - np.log(self.Z) - np.log(self.gamma**2 + x**2)

    def pdf(self,x):
        return np.exp(self.logpdf(x))

class Normal:
    def __init__(self,
                 mu,
                 std,
                unscaled = False):

        self.mu = mu
        self.std = std

        if unscaled:
            self.Z = 1.0
        else:
            self.Z = np.sqrt(2*np.pi)*self.std

    def rvs(self,N):
        return np.random.normal(self.mu,self.std,size = N)

    def logpdf(self,x):
        return -(x - self.mu)**2 / (2*self.std**2) - np.log(self.Z)

    def pdf(self,x):
        return np.exp(self.logpdf)


class ImportanceSampler:
    def __init__(self,
                 target_dist,
                 proposal_dist,
                ):

        self.target_dist = target_dist
        self.proposal_dist = proposal_dist

    def sample(self,N):
        x = self.proposal_dist.rvs(N)
        return x

    def compute_weights(self,x,inlog = False):
        w = self.target_dist.logpdf(x) - self.proposal_dist.logpdf(x)
        if not inlog:
             w = np.exp(w)
        return w

    def normalize_weights(self,):
        pass


