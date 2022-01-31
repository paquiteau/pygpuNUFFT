#!/usr/bin/env python

import argparse
from time import perf_counter
import numpy as np
# gpuNUFFT import
from mri.operators import NonCartesianFFT
from pysap.data import get_sample_data
from modopt.math.matrix import PowerMethod


parser = argparse.ArgumentParser(description='test density_compensation.')
parser.add_argument('dim', metavar='dim', type=int, const=0, default=0, nargs='?',
                    help='select dimension')

def test_powermethod(samples, shape, n_coils, smaps, density):
    f = NonCartesianFFT(samples=samples,
                        shape=shape,
                        density_comp=density,
                        n_coils=n_coils,
                        smaps=smaps,
                        implementation="gpuNUFFT",
                        )
    power = PowerMethod(lambda x: f.adj_op(f.op(x)),
                        data_shape=shape,
                        data_type="complex64",
                        auto_run=False,
                        verbose=True)
    ts = perf_counter()
    power.get_spec_rad(extra_factor=1.0, max_iter=10)
    rad = power.spec_rad
    tf = perf_counter()
    print("elapsed_time: {:.3f}s".format(tf-ts))
    print("rad: ", rad)

    ts2 = perf_counter()
    rad_new = f.impl.operator.get_spectral_radius(10, 1e-6)
    tf2 = perf_counter()
    print("elapsed_time: {:.3f}s, x{:.1f}".format(
        tf2-ts2, (tf-ts)/(tf2-ts2)))
    print("rad_new", rad_new)

if __name__ == "__main__":

    args = parser.parse_args()
    if args.dim == 2:
        img2d = get_sample_data("2d-pmri").data.astype("complex64")
        samples2d = get_sample_data("mri-radial-samples").data
        print(img2d.shape)
        samples2d *= np.pi / samples2d.max()
        shape2d = (512, 512)
        n_coils = 32
        smaps2d = np.ones_like(img2d, dtype='complex64')
        smaps2d = smaps2d * np.arange(1, len(smaps2d) + 1)[:, np.newaxis, np.newaxis]
        fake_data2d = np.ones(len(samples2d), dtype='complex64')
        img2dssos = np.linalg.norm(img2d,axis=0)
        density = np.ones(len(samples2d), dtype='float32')

        test_powermethod(samples2d, shape2d, n_coils, smaps2d, density)
    if args.dim == 3:
        img3d = get_sample_data("3d-pmri").data.astype(np.complex64)
        samples3d = get_sample_data("mri-radial-3d-samples").data
        samples3d *= np.pi / samples3d.max()
        n_coils = 32
        shape3d = (128, 128, 160)
        smaps3d = np.ones_like(img3d, dtype='complex128')
        smaps3d = smaps3d * np.arange(1, len(smaps3d) + 1)[:, np.newaxis, np.newaxis, np.newaxis]
        fake_data3d = np.ones(len(samples3d), dtype='complex64')
        img3dssos = np.linalg.norm(img3d, axis=0)
        density = np.ones(len(samples3d), dtype='float32')

        test_powermethod(samples3d, shape3d, n_coils, smaps3d, density)
