#!/usr/bin/env python

import argparse

import numpy as np
# gpuNUFFT import
from mri.operators import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation
from pysap.data import get_sample_data


parser = argparse.ArgumentParser(description='test density_compensation.')
parser.add_argument('dim', metavar='dim', type=int,
                    help='select dimension')


def test_density3D():
    print("# 3D DensityCompensation")
    samples3d = get_sample_data("mri-radial-3d-samples").data
    samples3d *= np.pi / samples3d.max()
    shape3d = (128, 128, 160)
    grid_op = NonCartesianFFT(
        samples=samples3d,
        shape=shape3d,
        implementation='gpuNUFFT',
        osf=1,
    )

    density_new = grid_op.impl.operator.estimate_density_comp(10)
    density_comp3d = estimate_density_compensation(samples3d, shape3d, 10)
    print(np.allclose(density_comp3d, density_new))


def test_density2D():
    print("# 2D DensityCompensation")
    samples2d = get_sample_data("mri-radial-samples").data
    samples2d *= 2 * np.pi
    shape2d = (512, 512)

    grid_op = NonCartesianFFT(
        samples=samples2d,
        shape=shape2d,
        implementation='gpuNUFFT',
        osf=1,
    )

    density_new = grid_op.impl.operator.estimate_density_comp(10)
    density_comp2d = estimate_density_compensation(samples2d, shape2d, 10)
    print(np.allclose(density_comp2d, density_new))


if __name__ == "__main__":
    args = parser.parse_args()

    if args.dim == 3:
        test_density3D()
    elif args.dim == 2:
        test_density2D()
    else:
        test_density2D()
        test_density3D()
