#!/usr/bin/env python

import argparse
from time import perf_counter
import numpy as np
# gpuNUFFT import
from mri.operators import NonCartesianFFT
from pysap.data import get_sample_data


parser = argparse.ArgumentParser(description='test gradient computations.')
parser.add_argument('dim', metavar='dim', type=int, const=0, default=0,
                    nargs='?', help='select dimension')
parser.add_argument('--new', action='store_true', default=False)
parser.add_argument('--gpu-density', dest='density',
                    action='store_true', default=False)


def test_gradient(image, shape, n_coils, samples, samples_data,
                  smaps=None, density=None, new=False):
    """Perform the gradient computation test."""
    fourier_op2 = NonCartesianFFT(samples=samples.copy(),
                                  shape=shape,
                                  density_comp=density.copy(),
                                  n_coils=n_coils,
                                  smaps=smaps,
                                  implementation="gpuNUFFT",
                                  )

    fourier_op = NonCartesianFFT(samples=samples.copy(),
                                 shape=shape,
                                 density_comp=density.copy(),
                                 n_coils=n_coils,
                                 smaps=smaps,
                                 implementation="gpuNUFFT",
                                 )
    start_t = perf_counter()
    grad = fourier_op2.adj_op(fourier_op2.op(
        image.copy()) - np.zeros_like(samples_data))
    end_t = perf_counter()
    print(f"elapsed_time: {end_t-start_t:.3f}s")
    print("||g|| = ", np.linalg.norm(grad.flatten()))
    if new:
        start_t2 = perf_counter()
        grad_new = fourier_op.impl.data_consistency(
            image.copy(), np.zeros_like(samples_data))
        end_t2 = perf_counter()
        print("elapsed_time: {:.3f}s, x{:.1f}".format(
            end_t2 - start_t2,
            (end_t - start_t) / (end_t2 - start_t2)))
        print("||g_new|| = ", np.linalg.norm(grad_new.flatten()))
        print("allclose: ", np.allclose(grad, grad_new))
        print("||g-g_new||/||g|| = ",
              np.linalg.norm(
                  (grad - grad_new).flatten()) / np.linalg.norm(grad))

        return grad.copy(), grad_new.copy()
    return grad.copy(), grad.copy()


def test_gradient_3d(new, density):
    """Setup the test for the 3D Gradient computation"""
    print("# 3D gradient")
    img3d = get_sample_data("3d-pmri").data.astype(np.complex64)
    samples3d = get_sample_data("mri-radial-3d-samples").data
    samples3d *= np.pi / samples3d.max()
    n_coils = 32
    shape3d = (128, 128, 160)
    smaps3d = np.ones_like(img3d, dtype='complex128')
    smaps3d = smaps3d * \
        np.arange(1, len(smaps3d) + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    fake_data3d = np.ones(len(samples3d), dtype='complex64')
    img3dssos = np.linalg.norm(img3d, axis=0)
    if density:
        grid_op = NonCartesianFFT(
            samples=samples3d,
            shape=shape3d,
            implementation="gpuNUFFT",
            osf=1,
        )

        density = grid_op.impl.operator.estimate_density_comp(10)
        del grid_op
    else:
        density = np.ones(len(samples3d), dtype='float32')

    return test_gradient(img3dssos, shape3d, n_coils, samples3d, fake_data3d,
                         smaps=smaps3d, density=density, new=new)


def test_gradient_2d(new, density):
    """Setup the test for the 2D Gradient computation"""
    print("# 2D data_consistency")
    img2d = get_sample_data("2d-pmri").data.astype(np.complex64)
    samples2d = get_sample_data("mri-radial-samples").data
    print(img2d.shape)
    samples2d *= np.pi / samples2d.max()
    shape2d = (512, 512)
    n_coils = 32
    smaps2d = np.ones_like(img2d, dtype='complex64')
    smaps2d = smaps2d * \
        np.arange(1, len(smaps2d) + 1)[:, np.newaxis, np.newaxis]
    fake_data2d = np.ones(len(samples2d), dtype='complex64')

    img2dssos = np.linalg.norm(img2d, axis=0)
    if density:
        grid_op = NonCartesianFFT(
            samples=samples2d,
            shape=shape2d,
            implementation="gpuNUFFT",
            osf=1,
        )

        density = grid_op.impl.operator.estimate_density_comp(10)
        del grid_op
    else:
        density = np.ones(len(samples2d), dtype='float32')

    return test_gradient(img2dssos, shape2d,
                         n_coils, samples2d,
                         fake_data2d, smaps=smaps2d,
                         density=density, new=new)


if __name__ == "__main__":

    args = parser.parse_args()

    if args.dim == 3:
        test_gradient_3d(args.new, args.density)
    elif args.dim == 2:
        test_gradient_2d(args.new, args.density)
    else:
        test_gradient_2d(args.new, args.density)
        test_gradient_3d(args.new, args.density)
