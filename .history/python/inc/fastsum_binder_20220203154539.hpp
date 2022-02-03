#ifndef FASTSUM_BINDER_HPP_INCLUDED
#define FASTSUM_BINDER_HPP_INCLUDED

#include "config.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef HAVE_COMPLEX_H
  #include <complex.h>
#endif


#include "fastsum.hpp"
#include "kernels.h"
#include "infft.h"

namespace py = pybind11;

class FastSumOperator
{
  int j, k; /**< indices                 */
  int d; /**< number of dimensions    */
  int N; /**< number of source nodes  */
  int M; /**< number of target nodes  */
  int n; /**< expansion degree        */
  int m; /**< cut-off parameter       */
  int p; /**< degree of smoothness    */
  const char *s; /**< name of kernel          */
  C (*kernel)(R, int, const R *); /**< kernel function         */
  R c; /**< parameter for kernel    */
  fastsum_plan my_fastsum_plan; /**< plan for fast summation */
  R time; /**< for time measurement    */
  R error = K(0.0); /**< for error computation   */
  R eps_I; /**< inner boundary          */
  R eps_B; /**< outer boundary          */


  public:
  FastSumOperator(int dimension, int N, int M, int n,  int m, int p, const char *s, R c, float eps_I, float eps_B);
  void init_random();
  py::array_t<R> sum(py::array_t<R>, bool);
 
  ~FastSumOperator()
  {    
  }

};


#endif // FASTSUM_BINDER_HPP_INCLUDED
