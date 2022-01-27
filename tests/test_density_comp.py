#!/usr/bin/env python

import argparse

from time import perf_counter
import numpy as np
# gpuNUFFT import
from mri.operators import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation
from pysap.data import get_sample_data


parser = argparse.ArgumentParser(description='test density_compensation.')
parser.add_argument('dim', metavar='dim', type=int, const=0,
                    default=0, nargs='?',
                    help='select dimension')
parser.add_argument('--iter', type=int, default=10, help="number of iteration")
parser.add_argument('--new', action='store_true', default=False, )




def test_density(samples, shape, new, itr=10):

    ts = perf_counter()
    density_comp = estimate_density_compensation(samples, shape, itr)
    tf = perf_counter()
    print("elapsed_time: {:.3f}s".format(tf-ts))
    print("||d||= ", np.linalg.norm(density_comp))
    if new:
        ts2 = perf_counter()
        grid_op = NonCartesianFFT(
                samples=samples,
                shape=shape,
                implementation="gpuNUFFT",
                osf=1,
            )
        density_new = grid_op.impl.operator.estimate_density_comp(itr)
        tf2 = perf_counter()
        print("elapsed_time: {:.3f}s, x{:.1f}".format(tf2-ts2, (tf-ts)/(tf2-ts2)))
        print("||d_new||= ", np.linalg.norm(density_new))
        print("allclose: ", np.allclose(density_comp, density_new))
        print("||d-d_new||/||d|| = ",np.linalg.norm(density_comp - density_new) / np.linalg.norm(density_comp))

def test_density3D(new, itr=10):
    print("# 3D DensityCompensation")
    samples3d = get_sample_data("mri-radial-3d-samples").data
    shape3d = (128, 128, 160)
    print("samples: ", samples3d.shape)
    print("img size ", shape3d)
    return test_density(samples3d, shape3d, new, itr)

def test_density2D(new, itr=10):
    print("# 2D DensityCompensation")
    samples2d = get_sample_data("mri-radial-samples").data
    shape2d = (512,512)
    print("samples: ", samples2d.shape)
    print("img size ", shape2d)
    return test_density(samples2d, shape2d, new, itr)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.dim == 3:
        test_density3D(args.new, itr=args.iter)
    elif args.dim == 2:
        test_density2D(args.new, itr=args.iter)
    else:
        test_density2D(args.new, itr=args.iter)
        test_density3D(args.new, itr=args.iter)
