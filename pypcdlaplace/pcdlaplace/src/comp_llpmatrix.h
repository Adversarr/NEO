#ifndef __COMP_LLPMATRIX_H__
#define __COMP_LLPMATRIX_H__

#include "datastructure.h"
#include "point_cloud.h"

//void generate_laplace_matrix_sparse_matlab_dim2(PCloud& pcloud, double h, double rho);
void generate_laplace_matrix_sparse_matlab_dim2(PCloud& pcloud, double h, double rho, vector<unsigned int>& II, vector<unsigned int>& JJ, vector<double>& SS);
void generate_laplace_matrix_sparse_matlab_dim3(PCloud& pcloud, double h, double rho, vector<unsigned int>& II, vector<unsigned int>& JJ, vector<double>& SS);

void generate_laplace_matrix_sparse_matlab_dimk(PCloud& pcloud, double h, double rho, unsigned int tdim, vector<unsigned int>& II, vector<unsigned int>& JJ, vector<double>& SS);

void generate_graph_laplace_matrix_sparse_matlab(PCloud& pcloud, double h, double rho, unsigned int tdim, vector<unsigned int>& II, vector<unsigned int>& JJ, vector<double>& SS);
void generate_arbdist_graph_laplace_matrix_sparse_matlab(PCloud& pcloud, double h, double rho, unsigned int tdim, vector<unsigned int>& II, vector<unsigned int>& JJ, vector<double>& SS);
void generate_kernel_matrix_sparse_matlab(PCloud& pcloud, double h, double rho, unsigned int tdim, vector<unsigned int>& II, vector<unsigned int>& JJ, vector<double>& SS);


//--------------------------------------------------------------------------------------------------------------//
//
//						Utility
//
//--------------------------------------------------------------------------------------------------------------//
void est_tangent_space_lapack(const dPoint& pt, const vector<dPoint>& neighbor_pts, double h, unsigned int tdim, vector<dVector>& tspace);
double simplex_sign_volume(vector<dPoint> points);

#endif //__COMP_LLPMATRIX_H__
