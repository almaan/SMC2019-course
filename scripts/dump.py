
for t in range(50):
    origin = origin[aidxs_apf[t,:]]
    opos = len(coords)
    for k,n in enumerate(aidxs_apf[t,:]):
        coords.append((t,xs_apf[t,k]))
        if len(coords) > n_particles:
            source =  opos - n_particles + int(n)
            target = opos + k
            edges.append((source,target))
            if origin[k] == survivor:
                sedges.append((source,target))



def run(self,ys,N):

        T = ys.shape[0]
        xt = self.x0_dist.rvs(N)

        ll = 0.0

        ws = self._omega(xt,ys[0],inlog = True)
        ll += np.log(np.exp(ws).sum()) - np.log(N)

        ws = expNormalize(ws,inlog = True)
        wss = np.zeros((N,T))
        wss[:,0] = ws

        xts = np.zeros((N,T))
        xts[:,0] = xt

        xt = self.resample(xt,ws,N)

        for t in range(1,T):
            xt = self.xt_xtm1_dist(xt).rvs()
            ws = self._omega(xt,ys[t],inlog = True)
            wss[:,t] = ws

            ll += np.log(np.exp(ws).sum()) - np.log(N)

            ws = expNormalize(ws,inlog = True)
            xt = self.resample(xt,ws,N)
            xts[:,t] = xt

        res = {'xs':xts,
               'weights':wss,
               'loglikelihood':ll,
              }

        return res


