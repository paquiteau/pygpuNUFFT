#!/usr/bin/env python

import argparse
from time import perf_counter
import numpy as np
# gpuNUFFT import
from mri.operators import NonCartesianFFT
from pysap.data import get_sample_data
from modopt.math.matrix import PowerMethod


def test_powermethod(samples, shape, n_coils, smaps, ):
    f = NonCartesianFFT(samples=samples,
                        shape=shape,
                        density_comp=density,
                        n_coils=n_coils,
                        smaps=smaps,
                        implementation="gpuNUFFT",
                        )
    power = PowerMethod(lambda x: f.adj_op(f.op(x)),
                        data_shape=shape,
                        data_type=np.complex64,
                        auto_run=False,
                        verbose=True)

    ts = perf_counter()
    rad = power.get_spec_rad(extra_factor=1, max_iter=10)
    tf = perf_counter()
    print("elapsed_time: {:.3f}s".format(tf-ts))
    print("rad", r)

    ts2 = perf_counter()
    r = f.impl.operator.get_spectral_radius(max_iter=10)
    tf2 = perf_counter()
    print("elapsed_time: {:.3f}s, x{:.1f}".format(
        tf2-ts2, (tf-ts)/(tf2-ts2)))
    print("rad_new", np.linalg.norm(gradient_new.flatten()))
