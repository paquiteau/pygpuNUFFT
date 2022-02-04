#include "fastsum_binder.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>


namespace py = pybind11;


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


FastSumOperator::FastSumOperator(int dimension, int N, int M, int n,  int m, int p, const char *s, R c, float eps_I=0.0625, float eps_B=0.0625)
{
  this->d = dimension;
  this->N = N;
  this->M = M;
  this->n = n;
  this->m = m;
  this->p = p;
  this->s = s;
  this->time = 0.0;
  this->error = 0.0;
  this->eps_I = eps_I;
  this->eps_B = eps_B;
  this->c = c;
  kernel_fs my_kernel;
  get_kernel_from_name(s, my_kernel);
  fastsum_init_guru(&(this->my_fastsum_plan), d, N, M, my_kernel, &this->c, 0, n, m, p, eps_I, eps_B);

}

void FastSumOperator::init_random()
{
  /** init source knots in a d-ball with radius 0.25-eps_b/2 */
  k = 0;
  while (k < M)
  {
    R r_max = K(0.25) - my_fastsum_plan.eps_B / K(2.0);
    R r2 = K(0.0);

    for (j = 0; j < d; j++)
      my_fastsum_plan.x[k * d + j] = K(2.0) * r_max * 0.08 - r_max;

    for (j = 0; j < d; j++)
      r2 += my_fastsum_plan.x[k * d + j] * my_fastsum_plan.x[k * d + j];

    if (r2 >= r_max * r_max)
      continue;

    k++;
  }
  
  for (k = 0; k < N; k++)
  {
    my_fastsum_plan.alpha[k] = 1;
  }
  /** init target knots in a d-ball with radius 0.25-eps_b/2 */
  k = 0;
  while (k < M)
  {
    R r_max = K(0.25) - my_fastsum_plan.eps_B / K(2.0);
    R r2 = K(0.0);

    for (j = 0; j < d; j++)
      my_fastsum_plan.y[k * d + j] = K(2.0) * r_max * 0.01  - r_max;

    for (j = 0; j < d; j++)
      r2 += my_fastsum_plan.y[k * d + j] * my_fastsum_plan.y[k * d + j];

    if (r2 >= r_max * r_max)
      continue;

    k++;
  }
}

py::array_t<R> FastSumOperator::sum(py::array_t<R> points, py::array_t<R> points2, bool direct)
{
  // Make all alphas as ones for first set of potentials
  k = 0;
  for (k = 0; k < N; k++)
  {
    my_fastsum_plan.alpha[k] = (R) 1;
  }
  // Initialize all the source knots
  R *data = (R *) points.request().ptr;
  R *data2 = (R *) points2.request().ptr;
  for (k = 0; k < N*d; k++)
  {
    my_fastsum_plan.x[k] = data[k];
    my_fastsum_plan.y[k] = data[k];  // FIXME, this is not memory efficient
    my_fastsum_plan.x2[k] = data2[k];
    my_fastsum_plan.y2[k] = data2[k];
    if(k<N)
      my_fastsum_plan.alpha[k] = (R) 1;
  }
  if(direct)
  { 
    fastsum_exact(&my_fastsum_plan);
  }
  else
  {
    // precomputation 
    fastsum_precompute(&my_fastsum_plan);
    fastsum_trafo(&my_fastsum_plan);
  }
  
  /** copy result */
  py::array_t<R> out_result({my_fastsum_plan.M_total});
  R *out_data = (R *) out_result.request().ptr;
  
  for (j = 0; j < my_fastsum_plan.M_total; j++)
    out_data[j] = my_fastsum_plan.f[j].real();
  
  return out_result;
}
