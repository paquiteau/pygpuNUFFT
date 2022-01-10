#!/usr/bin/env python

from time import sleep
import numpy as np
# gpuNUFFT import
from mri.operators import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation
from pysap.data import get_sample_data


img2d = get_sample_data("2d-pmri").data.astype(np.complex64)
samples2d = get_sample_data("mri-radial-samples").data
samples2d *= 2 * np.pi
shape2d = (512, 512)
n_samples2d = 32768
img2dssos = np.linalg.norm(img2d,axis=0).astype(np.complex64)


density2d = estimate_density_compensation(samples2d, shape2d)

fake_data2d = np.ones(shape=len(samples2d), dtype="complex64")
gpuNUFFT = NonCartesianFFT(samples=samples2d,
                            shape=shape2d,
                            n_coils=n_coils,
                            smaps=smaps2d,
                            density_comp=density2d,
                            implementation='gpuNUFFT')

gradient_fast = gpuNUFFT.impl.operator.data_consistency(img2dssos,fake_data2d)

gradient = gpuNUFFT.adj_op(gpuNUFFT.op(img2dssos)-fake_data2d)




img3d = get_sample_data("3d-pmri").data.astype(np.complex64)
samples3d = get_sample_data("mri-radial-3d-samples").data
samples3d *= np.pi / samples3d.max()

smaps3d = np.ones_like(img3d, dtype='complex128')
smaps3d = smaps3d * np.arange(1, len(smaps3d) + 1)[:, np.newaxis, np.newaxis, np.newaxis]
n_coils = 32
n_samples3d = 6136781
shape3d = (128, 128, 160)
img3dssos = np.linalg.norm(img3d,axis=0).astype(np.complex64)

density_comp3d = estimate_density_compensation(samples3d, shape3d)

fake_data = np.ones(shape=len(samples3d), dtype="complex64")
print('sense')
gpuNUFFT = NonCartesianFFT(samples=samples3d,
                            shape=shape3d,
                            n_coils=n_coils,
                            smaps=smaps3d,
                            density_comp=density_comp3d,
                            implementation='gpuNUFFT')

gradient_fast = gpuNUFFT.impl.operator.data_consistency(img3dssos,fake_data)

gradient = gpuNUFFT.adj_op(gpuNUFFT.op(img3dssos)-fake_data)
print('gradient done')

print(np.allclose(gradient_fast, gradient.T))
print(np.linalg.norm(gradient_fast-gradient.T))


print('calibrationless')
gpuNUFFT = NonCartesianFFT(samples=samples3d,
                            shape=shape3d,
                            n_coils=n_coils,
                            density_comp=density_comp3d,
                            implementation='gpuNUFFT')
kspace3d = gpuNUFFT.op(img3d)
img_autoadj = gpuNUFFT.adj_op(kspace3d)
del gpuNUFFT
