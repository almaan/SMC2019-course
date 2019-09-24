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

def main(n_particles = 100, T = 250):

    ts, xs, ys = load_generated_traj()
    apf = make_apf_obj()

    res = apf.run(n_particles,ys,)

    xs_apf = res['xts']
    aidx_apf = res['aidxs']

    long_res = to_long_format(xs_apf,
                              aidx_apf,
                              T = T,
                              )

    fig, ax = visualize_genealogy(**long_res,
                                  window = 5)

    odir = osp.join(osp.dirname(osp.dirname(osp.abspath(__name__))),'res')

    if not osp.exists(odir):
        mkdir(odir)

    fig.savefig(osp.join(odir,'ex2-gen-exe.png'))

if __name__ == '__main__':
    main()

