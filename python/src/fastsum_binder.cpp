#include "fastsum_binder.hpp"


void get_kernel_from_name(const char *s, kernel_fs &my_kernel)
{
    if (strcmp(s, "gaussian") == 0)
      my_kernel = gaussian;
    else if (strcmp(s, "multiquadric") == 0)
      my_kernel = multiquadric;
    else if (strcmp(s, "inverse_multiquadric") == 0)
      my_kernel = inverse_multiquadric;
    else if (strcmp(s, "logarithm") == 0)
      my_kernel = logarithm;
    else if (strcmp(s, "thinplate_spline") == 0)
      my_kernel = thinplate_spline;
    else if (strcmp(s, "one_over_square") == 0)
      my_kernel = one_over_square;
    else if (strcmp(s, "one_over_modulus") == 0)
      my_kernel = one_over_modulus;
    else if (strcmp(s, "one_over_x") == 0)
      my_kernel = one_over_x;
    else if (strcmp(s, "inverse_multiquadric3") == 0)
      my_kernel = inverse_multiquadric3;
    else if (strcmp(s, "sinc_kernel") == 0)
      my_kernel = sinc_kernel;
    else if (strcmp(s, "cosc") == 0)
      my_kernel = cosc;
    else if (strcmp(s, "cot") == 0)
      my_kernel = kcot;
    else if (strcmp(s, "one_over_cube") == 0)
      my_kernel = one_over_cube;
    else if (strcmp(s, "log_sin") == 0)
      my_kernel = log_sin;
    else if (strcmp(s, "laplacian_rbf") == 0)
      my_kernel = laplacian_rbf;
    else
    {
      my_kernel = multiquadric;
    }
}


FastSumOperator::FastSumOperator(int dimension, int N, int M, int n,  int m, const char *s)
{
    this->d = dimension;
    this->N = N;
    this->M = M;
    this->n = n;
    this->m = m;
    this->p = m;
    this->s = s;
    this->time = 0.0;
    this->error = 0.0;
    this->eps_I = 1/32;
    this->eps_B = 1/32;
    kernel_fs my_kernel;
    get_kernel_from_name(s, my_kernel);
    fastsum_init_guru(&(this->my_fastsum_plan), d, N, M, my_kernel, &this->c, 0, n, m, p, eps_I, eps_B);
}


