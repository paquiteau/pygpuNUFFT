import matplotlib.pyplot as plt
import numpy as np
from gpuNUFFT import FastSumOp
from time import perf_counter
from sparkling.fmm.gpu.keops import direct_pykeops_compute_torch
#r::FastSumOperator(int dimension, int N, int M, int n,  int m, int p, const char *s, R c, float eps_I=0.0625, float eps_B=0.0625)
"""    
    printf("\nfastsum_test d N M n m p kernel c eps_I eps_B\n\n");
    printf("  d       dimension                 \n");
    printf("  N       number of source nodes    \n");
    printf("  M       number of target nodes    \n");
    printf("  n       expansion degree          \n");
    printf("  m       cut-off parameter         \n");
    printf("  p       degree of smoothness      \n");
    printf("  kernel  kernel function  (e.g., gaussian)\n");
    printf("  c       kernel parameter          \n");
    printf("  eps_I   inner boundary            \n");
    printf("  eps_B   outer boundary            \n\n");
"""
np.random.seed(0)
times_keops = []
times_fastsum = []
for N in np.logspace(3, 8, 10)[:1]:
    dimension = 3
    N = int(N)
    fastsum = FastSumOp(dimension, N, N, 128, 4, 2, "inverse_multiquadric", 1e-5, 0.003125, 0.003125)
    max_val = (1/4-0.003125)/np.sqrt(dimension)
    points = np.random.uniform(-max_val,  max_val, (N, dimension))
    print(points[0])
    st = perf_counter()
    pots = fastsum.sum(points, False)
    times_fastsum.append(perf_counter() - st)
    print("Time for fastsum : ", perf_counter() - st)
    st = perf_counter()
    pots_keops = direct_pykeops_compute_torch(points)
    times_keops.append(perf_counter() - st)
    print("Time for keops : ", perf_counter() - st)
    print(points[0])
times_fastsum