//#include <mex.h>
#include <math.h>
#include <vector>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>

#include "point_cloud.h"
#include "comp_llpmatrix.h"

#include <ctime>

int main(int argc, char* argv[])
{
  
  PCloud pcloud;
  printf("Try to load file %s\n", argv[1]);
  int np = pcloud.ReadFromPCD(argv[1]);//np < 0 if error 

  int tdim = 2, dim = 3;	
  unsigned int nn = 10, htype = 0;
  double hs = 2, rho = 3;

  printf("np: %d, dim: %d, tdim: %d, ", np, dim, tdim);
  printf("htype: %d, nn: %d, hs: %.2f, rho: %.2f, ", htype, nn, hs, rho);

  double h;

  double avers = pcloud.average_size(nn);

  if(htype == 0)
  {
    h = avers * hs;
  }
  else
  {
    h = hs;
  }

  printf("avers: %f, h: %f\n", avers, h);

  vector<double> SSV;
  vector<unsigned int> IIV, JJV;

  time_t tstart, tend;
  tstart = time(0);
  generate_laplace_matrix_sparse_matlab_dim2(pcloud, h, rho, IIV, JJV, SSV);
  tend = time(0);

  cout << "It took "<< difftime(tend, tstart) <<" second(s)."<< endl;

  unsigned int nelem = IIV.size();
  if( nelem != JJV.size() ||  nelem != SSV.size() ){
      fprintf(stderr, "Dimension of II, JJ, SS has to be the same");
  }

  FILE* out_file;
  out_file = fopen("out.txt", "w");
  for(unsigned int i=0; i< nelem; ++i)
    {
      fprintf(out_file, "%d %d %lf\n", IIV[i], JJV[i], SSV[i]);
    }
  fclose(out_file);
  return 0;
}
