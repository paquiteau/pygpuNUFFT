#!/usr/bin/env python
import numpy as np
# gpuNUFFT import
from mri.operators import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation
from pysap.data import get_sample_data
import time

# Density Compensation 2D

# img2d = get_sample_data("2d-pmri").data.astype(np.complex64)
# samples2d = get_sample_data("mri-radial-samples").data
# samples2d *= 2 * np.pi
# shape2d = (512, 512)
# n_samples2d = 32768
# img2dssos = np.linalg.norm(img2d,axis=0).astype(np.complex64)

# first_op = NonCartesianFFT(samples=samples2d,
#                            shape=shape2d,
#                            implementation='gpuNUFFT'
#                            )


# grid_op = NonCartesianFFT(
#     samples=samples2d,
#     shape=shape2d,
#     implementation='gpuNUFFT',
#     osf=1,
# )
# density2d_new = grid_op.impl.operator.estimate_density_comp(10)
# # density2d = estimate_density_compensation(samples2d, shape2d, 10)


# # print(np.allclose(density2d, density2d_new))
# # print(np.linalg.norm(density2d_new-density2d))
# # print(density2d_new.real)
# # print(density2d)

# sleep(2)


# Density Compensation 3d
img3d = get_sample_data("3d-pmri").data.astype(np.complex64)
samples3d = get_sample_data("mri-radial-3d-samples").data
samples3d *= 2 * np.pi
shape3d = (128, 128, 160)
n_samples3d = 6136781
img2dssos = np.linalg.norm(img3d,axis=0).astype(np.complex64)

s = time.perf_counter()

grid_op = NonCartesianFFT(
        samples=samples3d,
        shape=shape3d,
        implementation='gpuNUFFT',
        osf=1,
)
density3d_new = grid_op.impl.operator.estimate_density_comp(10)
del grid_op

print("first run:", s-time.perf_counter())

s = time.perf_counter()
grid_op = NonCartesianFFT(
        samples=samples3d,
        shape=shape3d,
        implementation='gpuNUFFT',
        osf=1,
)
density3d_new = grid_op.impl.operator.estimate_density_comp(20)
del grid_op
print("second run:", s-time.perf_counter())

s = time.perf_counter()
density3d = estimate_density_compensation(samples3d, shape3d, 10)
print("third run:", s-time.perf_counter())


print(np.allclose(density3d, density3d_new))
print(np.linalg.norm(density3d_new-density3d))
