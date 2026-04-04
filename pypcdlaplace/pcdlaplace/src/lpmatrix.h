#ifndef __LPMATRIX_H__
#define __LPMATRIX_H__

#include <cstddef>
#include <string>
#include <vector>
using namespace std;

namespace pcdlaplace {

enum class HType {
  DataDriven = 0,
  PreSpecified = 1
};

struct Options {
  unsigned int nn;
  double hs;
  double rho;
  HType htype;

  Options() : nn(10), hs(2.0), rho(3.0), htype(HType::DataDriven) {}
};

struct SparseTriplets {
  vector<size_t> rows;
  vector<size_t> cols;
  vector<double> values;
  size_t nrows;
  size_t ncols;
  double h;

  SparseTriplets() : nrows(0), ncols(0), h(0.0) {}
};

HType parse_htype(const string& htype);

SparseTriplets compute_pcdlaplace_matrix_sparse(const double *points,
                                                size_t np,
                                                size_t dim,
                                                size_t tdim,
                                                const Options& options);

}  // namespace pcdlaplace

void generate_pcdlaplace_matrix_sparse_matlab(double *points, unsigned int np, unsigned int dim, unsigned int tdim, unsigned int htype, unsigned int nn, double hs, double rho, vector<unsigned int>& IIV, vector<unsigned int>& JJV, vector<double>& SSV);

void generate_graphlaplace_matrix_sparse_matlab(double *points, unsigned int np, unsigned int dim, unsigned int tdim, unsigned int htype, unsigned int nn, double hs, double rho, vector<unsigned int>& IIV, vector<unsigned int>& JJV, vector<double>& SSV);

void generate_kernelmatrix_sparse_matlab(double *points, unsigned int np, unsigned int dim, unsigned int tdim, unsigned int htype, unsigned int nn, double hs, double rho, vector<unsigned int>& IIV, vector<unsigned int>& JJV, vector<double>& SSV, double& h);

void generate_arbdistgraphlaplace_matrix_sparse_matlab(double *points, unsigned int np, unsigned int dim, unsigned int tdim, unsigned int htype, unsigned int nn, double hs, double rho, vector<unsigned int>& IIV, vector<unsigned int>& JJV, vector<double>& SSV);

#endif //__LPMATRIX_H__
