//#include <mex.h>
#include <limits>
#include <stdexcept>

#include "comp_llpmatrix.h"
#include "lpmatrix.h"

namespace {

struct LegacySparseResult {
  vector<unsigned int> rows;
  vector<unsigned int> cols;
  vector<double> values;
  double h;

  LegacySparseResult() : h(0.0) {}
};

void validate_inputs(const double *points,
                     size_t np,
                     size_t dim,
                     size_t tdim,
                     const pcdlaplace::Options& options)
{
  if (points == NULL)
    {
      throw std::invalid_argument("Point buffer must not be null");
    }
  if (np == 0 || dim == 0)
    {
      throw std::invalid_argument("Point cloud must have shape (n, d) with n > 0 and d > 0");
    }
  if (tdim == 0 || tdim > dim)
    {
      throw std::invalid_argument("tdim must satisfy 1 <= tdim <= ambient dimension");
    }
  if (options.nn <= 1)
    {
      throw std::invalid_argument("nn must be greater than 1");
    }
  if (options.hs <= 0.0)
    {
      throw std::invalid_argument("hs must be positive");
    }
  if (options.rho <= 0.0)
    {
      throw std::invalid_argument("rho must be positive");
    }
  if (np < tdim + 1)
    {
      throw std::invalid_argument("Point cloud is too small for the requested tangent dimension");
    }
  if (np > static_cast<size_t>(numeric_limits<unsigned int>::max()))
    {
      throw std::invalid_argument("Point cloud is too large for the legacy core");
    }
}

LegacySparseResult compute_legacy_sparse(const double *points,
                                         unsigned int np,
                                         unsigned int dim,
                                         unsigned int tdim,
                                         const pcdlaplace::Options& options)
{
  PCloud pcloud(points, np, dim);
  const double avers = pcloud.average_size(options.nn);
  const double h = (options.htype == pcdlaplace::HType::DataDriven) ? (avers * options.hs)
                                                                    : options.hs;

  LegacySparseResult result;
  result.h = h;
  if (tdim == 2)
    {
      generate_laplace_matrix_sparse_matlab_dim2(pcloud, h, options.rho, result.rows, result.cols, result.values);
    }
  else if (tdim == 3)
    {
      generate_laplace_matrix_sparse_matlab_dim3(pcloud, h, options.rho, result.rows, result.cols, result.values);
    }
  else
    {
      generate_laplace_matrix_sparse_matlab_dimk(pcloud, h, options.rho, tdim, result.rows, result.cols, result.values);
    }

  if (result.rows.size() != result.cols.size() || result.rows.size() != result.values.size())
    {
      throw std::runtime_error("Sparse triplet output is inconsistent");
    }
  return result;
}

}  // namespace

namespace pcdlaplace {

HType
parse_htype(const string& htype)
{
  if (htype == "ddr")
    {
      return HType::DataDriven;
    }
  if (htype == "psp")
    {
      return HType::PreSpecified;
    }
  throw std::invalid_argument("htype must be either 'ddr' or 'psp'");
}

SparseTriplets
compute_pcdlaplace_matrix_sparse(const double *points,
                                 size_t np,
                                 size_t dim,
                                 size_t tdim,
                                 const Options& options)
{
  validate_inputs(points, np, dim, tdim, options);
  const LegacySparseResult legacy = compute_legacy_sparse(points,
                                                          static_cast<unsigned int>(np),
                                                          static_cast<unsigned int>(dim),
                                                          static_cast<unsigned int>(tdim),
                                                          options);

  SparseTriplets result;
  result.nrows = np;
  result.ncols = np;
  result.h = legacy.h;
  result.rows.reserve(legacy.rows.size());
  result.cols.reserve(legacy.cols.size());
  result.values = legacy.values;

  for (size_t i = 0; i < legacy.rows.size(); ++i)
    {
      if (legacy.rows[i] == 0 || legacy.cols[i] == 0)
        {
          throw std::runtime_error("Legacy core returned invalid 1-based indices");
        }
      result.rows.push_back(static_cast<size_t>(legacy.rows[i] - 1));
      result.cols.push_back(static_cast<size_t>(legacy.cols[i] - 1));
    }
  return result;
}

}  // namespace pcdlaplace


void generate_pcdlaplace_matrix_sparse_matlab(double *points, unsigned int np, unsigned int dim, unsigned int tdim, unsigned int htype, unsigned int nn, double hs, double rho, vector<unsigned int>& IIV, vector<unsigned int>& JJV, vector<double>& SSV)
{
  pcdlaplace::Options options;
  options.nn = nn;
  options.hs = hs;
  options.rho = rho;
  options.htype = (htype == 0) ? pcdlaplace::HType::DataDriven : pcdlaplace::HType::PreSpecified;

  const LegacySparseResult result = compute_legacy_sparse(points, np, dim, tdim, options);
  IIV = result.rows;
  JJV = result.cols;
  SSV = result.values;
}

void generate_graphlaplace_matrix_sparse_matlab(double *points, unsigned int np, unsigned int dim, unsigned int tdim, unsigned int htype, unsigned int nn, double hs, double rho, vector<unsigned int>& IIV, vector<unsigned int>& JJV, vector<double>& SSV)
{
	PCloud pcloud(points, np, dim);

	//--------------------------------------------------
   //ofstream fout; 
	//fout.open("points");
	//for(unsigned int i = 0; i < np; i ++){ 
   //	for(unsigned int j = 0; j < dim; j ++){ 
	//  	fout<< points[j * np + i] <<" ";
	//	}
	//	fout<<endl;
	//}
	//pcloud.OutPCloud("pcd");
	//--------------------------------------------------
	double h;
	double avers = pcloud.average_size(nn);
	printf("avers: %f\n", avers);
	if(htype == 0){
		h = avers * hs;
	}
	else{
		h = hs;
	}

	printf("h: %f\n", h);
	generate_graph_laplace_matrix_sparse_matlab(pcloud, h, rho, tdim, IIV, JJV, SSV);
}

void generate_kernelmatrix_sparse_matlab(double *points, unsigned int np, unsigned int dim, unsigned int tdim, unsigned int htype, unsigned int nn, double hs, double rho, vector<unsigned int>& IIV, vector<unsigned int>& JJV, vector<double>& SSV, double& h)
{
	PCloud pcloud(points, np, dim);

	//--------------------------------------------------
   //ofstream fout; 
	//fout.open("points");
	//for(unsigned int i = 0; i < np; i ++){ 
   //	for(unsigned int j = 0; j < dim; j ++){ 
	//  	fout<< points[j * np + i] <<" ";
	//	}
	//	fout<<endl;
	//}
	//pcloud.OutPCloud("pcd");
	//--------------------------------------------------
	double avers = pcloud.average_size(nn);
	printf("avers: %f\n", avers);
	if(htype == 0){
		h = avers * hs;
	}
	else{
		h = hs;
	}

	printf("h: %f\n", h);
	generate_kernel_matrix_sparse_matlab(pcloud, h, rho, tdim, IIV, JJV, SSV);
}

void generate_arbdistgraphlaplace_matrix_sparse_matlab(double *points, unsigned int np, unsigned int dim, unsigned int tdim, unsigned int htype, unsigned int nn, double hs, double rho, vector<unsigned int>& IIV, vector<unsigned int>& JJV, vector<double>& SSV)
{
	PCloud pcloud(points, np, dim);

	//--------------------------------------------------
   //ofstream fout; 
	//fout.open("points");
	//for(unsigned int i = 0; i < np; i ++){ 
   //	for(unsigned int j = 0; j < dim; j ++){ 
	//  	fout<< points[j * np + i] <<" ";
	//	}
	//	fout<<endl;
	//}
	//pcloud.OutPCloud("pcd");
	//--------------------------------------------------
	double h;
	double avers = pcloud.average_size(nn);
	printf("avers: %f\n", avers);
	if(htype == 0){
		h = avers * hs;
	}
	else{
		h = hs;
	}

	printf("h: %f\n", h);
	generate_arbdist_graph_laplace_matrix_sparse_matlab(pcloud, h, rho, tdim, IIV, JJV, SSV);
}
