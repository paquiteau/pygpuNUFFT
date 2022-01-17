#!/usr/bin/env python

import argparse

import numpy as np
# gpuNUFFT import
from mri.operators import NonCartesianFFT
from pysap.data import get_sample_data


parser = argparse.ArgumentParser(description='test density_compensation.')
parser.add_argument('dim', metavar='dim', type=int,
                    help='select dimension')


def test_gradient(image, shape, n_coils, samples, samples_data, smaps=None, density=None):
    fourier_op = NonCartesianFFT(samples=samples,
                                 shape=shape,
                                 density_comp=density,
                                 n_coils=n_coils,
                                 smaps=smaps,
                                 implementation="gpuNUFFT",
                                 )
    print("fourier_op init done")
    gradient = fourier_op.adj_op(fourier_op.op(image) - samples_data)
    print("gradient classic done")
    gradient_new = fourier_op.impl.operator.data_consistency(
        image, samples_data)
    print("gradient new done")
    print(np.allclose(gradient, gradient_new.T))


def test_gradient3D():
    print("# 3D data_consistency")
    img3d = get_sample_data("3d-pmri").data.astype(np.complex64)
    samples3d = get_sample_data("mri-radial-3d-samples").data
    samples3d *= np.pi / samples3d.max()
    n_coils = 32
    shape3d = (128, 128, 160)
    smaps3d = np.ones_like(img3d, dtype='complex128')
    smaps3d = smaps3d * np.arange(1, len(smaps3d) + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    fake_data3d = np.ones(len(samples3d), dtype='complex64')
    grid_op = NonCartesianFFT(
        samples=samples3d,
        shape=shape3d,
        implementation='gpuNUFFT',
        osf=1,
    )

    img3dssos = np.linalg.norm(img3d)
    density = grid_op.impl.operator.estimate_density_comp(10)
    del grid_op

    return test_gradient(img3dssos, shape3d, n_coils, samples3d, fake_data3d, smaps=smaps3d, density=density)


def test_gradient2D():
    print("# 2D data_consistency")
    img2d = get_sample_data("2d-pmri").data.astype(np.complex64)
    samples2d = get_sample_data("mri-radial-samples").data
    print(img2d.shape)
    samples2d *= np.pi / samples2d.max()
    shape2d = (512, 512)
    n_coils = 32
    smaps2d = np.ones_like(img2d, dtype='complex64')
    smaps2d = smaps2d * np.arange(1, len(smaps2d) + 1)[:, np.newaxis, np.newaxis]
    fake_data2d = np.ones(len(samples2d), dtype='complex64')
    grid_op = NonCartesianFFT(
        samples=samples2d,
        shape=shape2d,
        implementation='gpuNUFFT',
        osf=1,
    )
    img2dssos = np.linalg.norm(img2d)
    density = grid_op.impl.operator.estimate_density_comp(10)
    del grid_op

    return test_gradient(img2dssos, shape2d,
                         n_coils, samples2d,
                         fake_data2d, smaps=smaps2d,
                         density=density)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.dim == 3:
        test_gradient3D()
    elif args.dim == 2:
        test_gradient2D()
    else:
        test_gradient2D()
        test_gradient3D()
