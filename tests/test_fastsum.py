import numpy as np
from gpuNUFFT import FastSumOp


a = FastSumOp(3, 1000, 1000, 128, 1, "gaussian")
print(a)