#!/usr/bin/env python3

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

#sigma2_a = 0.5
#sigma2_b = 0.01
#
#xtm1 = 0.16
#xt = 0.12
#yt = 0.3
#mean_a = 0.8*xtm1
#phi = 2
#
#
#sigma2 = 1/(sigma2_a + sigma2_b)
#p2 = 1/sigma2_b*yt + 1/sigma2_a*phi*xtm1
#dahln = st.norm(sigma2 * p2, sigma2**0.5)
#
#
#mu = np.array([mean_a,mean_a*phi])
#cov = np.array([[sigma2_a,sigma2_a*phi],[sigma2_a*phi,sigma2_b + phi**2*sigma2_a]])
#print(mu)
#print(cov)
#me = st.multivariate_normal(mu,cov)
#
#print(dahln.logpdf(xt))
#print(me.logpdf(np.array([xt,yt])))

#xtm1 = 0.32
#
#xt = np.random.normal(0.8*xtm1,np.sqrt(0.5),size = 100000)
#yt1 = np.random.normal(2*xt,np.sqrt(0.01))
#yt2 = np.random.normal(1.6*xtm1, np.sqrt(2+ 0.01),100000)
#
#print(yt2.mean())
#print(yt1.mean())
#print(yt1.var())
#print(yt2.var())
#
#plt.hist(yt1,color = 'red',alpha = 0.3,density = True)
#plt.hist(yt2,color = 'blue',alpha = 0.3,density = True)
#plt.show()


import numpy
import matplotlib.pyplot as plt

points = numpy.array([[1,2],[4,5],[2,7],[3,9],[9,2]])
edges = numpy.array([[0,1],[3,4],[3,2],[2,4]])

x = points[:,0].flatten()
y = points[:,1].flatten()

plt.plot(x[edges.T], y[edges.T], linestyle='-', color='y',
                 markerfacecolor='red', marker='o')

plt.show()
