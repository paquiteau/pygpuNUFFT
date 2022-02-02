#include "fastsum_binder.hpp"

FastSumOperator::FastSumOperator(int dimension, int N, int M, int n,  int m, const char *s)
{
    this->d = dimension;
    this->N = N;
    this->M = M;
    this->n = n;
    this->m = m;
    this->p = m;
    this->s = s;
    this->my_fastsum_plan = NULL;
    this->time = 0.0;
    this->error = 0.0;
    this->eps_I = 0.0;
    this->eps_B = 0.0;
}