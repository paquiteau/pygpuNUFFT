#!/usr/env/bin python

import numpy as np
# gpuNUFFT import
from mri.operators import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation
from pysap.data import get_sample_data


img2d = get_sample_data("2d-pmri").data.astype(np.complex64)
img3d = get_sample_data("3d-pmri").data.astype(np.complex64)
samples3d = get_sample_data("mri-radial-3d-samples").data
samples2d = get_sample_data("mri-radial-samples").data

samples2d *= 2 * np.pi
samples3d *= np.pi / samples3d.max()

smaps3d = np.ones_like(img3d, dtype='complex128')
smaps3d = smaps3d * np.arange(1, len(smaps3d) + 1)[:, np.newaxis, np.newaxis, np.newaxis]
n_coils = 32
shape2d = (512, 512)
n_samples2d = 32768
n_samples3d = 6136781
shape3d = (128, 128, 160)

for i in range(1):
    density_comp3d = estimate_density_compensation(samples3d, shape3d)
    print('density  comp done')
    gpuNUFFT = NonCartesianFFT(samples=samples3d,
                               shape=shape3d,
                               n_coils=n_coils,
                               smaps=smaps3d,
                               density_comp=density_comp3d,
                               implementation='gpuNUFFT')
    kspace3d = gpuNUFFT.op(img3d)
    img_autoadj = gpuNUFFT.adj_op(kspace3d)
    del gpuNUFFT
