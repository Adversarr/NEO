#include <mex.h>
#include <math.h>
#include <vector>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>

#include "comp_llpmatrix.h"


using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/*  check for proper number of arguments */
	/* NOTE: You do not need an else statement when using mexErrMsgTxt
	 *       within an if statement, because it will never get to the else
	 *       statement if mexErrMsgTxt is executed. (mexErrMsgTxt breaks you out of
	 *       the MEX-file) 
	*/
	if(nrhs != 2){ 
		mexErrMsgTxt("2 inputs required.");
	}

	int tdim = 2, dim = 3;	
	unsigned int nn = 10, htype = 0;
	double hs = 2, rho = 3;
	
	// Parse the config options
	if(nrhs == 2){
		const mxArray *opt = prhs[1];
		mxClassID category = mxGetClassID(opt);
		if(category != mxSTRUCT_CLASS){
			mexErrMsgTxt("Third input must be a structure");
		}
	 	mwSize total_num_of_elements;
  		mwIndex index;
  		int number_of_fields, field_index;
  		const char  *field_name;
  		const mxArray *field_array_ptr;
  		total_num_of_elements = mxGetNumberOfElements(opt); 
  		number_of_fields = mxGetNumberOfFields(opt);
  
  		/* Walk through each structure element. */
  		for (index=0; index<total_num_of_elements; index++)  {
    
    		/* For the given index, walk through each field. */ 
    		for (field_index=0; field_index<number_of_fields; field_index++)  {
         	field_name = mxGetFieldNameByNumber(opt, field_index);
				field_array_ptr = mxGetFieldByNumber(opt, index, field_index);
		      if (field_array_ptr == NULL) {
					continue;
				}

				if(strcmp(field_name, "nn") == 0){
					mxClassID   cat = mxGetClassID(field_array_ptr);
					if(cat == mxDOUBLE_CLASS){
						//mexPrintf("nn: %d", nn);
						nn = (unsigned int)(*mxGetPr(field_array_ptr));
					}
      		} 			
				else if(strcmp(field_name, "hs") == 0){
					mxClassID   cat = mxGetClassID(field_array_ptr);
					if(cat == mxDOUBLE_CLASS){
						//mexPrintf("hs: %f", hs);
						hs = (*mxGetPr(field_array_ptr));
					}
				}
				else if(strcmp(field_name, "rho") == 0){
					mxClassID   cat = mxGetClassID(field_array_ptr);
					if(cat == mxDOUBLE_CLASS){
						//mexPrintf("rho: %f", rho);
						rho = (*mxGetPr(field_array_ptr));
					}
				}
				else if(strcmp(field_name, "htype") == 0){
					mxClassID   cat = mxGetClassID(field_array_ptr);
					if(cat == mxCHAR_CLASS){
						char buf[256];
    					mwSize buflen;

		    			/* Allocate enough memory to hold the converted string. */
				    	buflen = mxGetNumberOfElements(field_array_ptr) + 1;
						if(buflen > 256){
							buflen = 256;
						}

					  	/* Copy the string data from string_array_ptr and place it into buf. */
				  	  	if (mxGetString(field_array_ptr, buf, buflen) == 0){
					  		if( strcmp(buf, "ddr") == 0 ){
								htype = 0;	
							}
							else if( strcmp(buf, "psp") == 0 ){
								htype = 1;
							}
					  	}
					}
				}
			}// for field_index
      }//for index
   }//if(nrhs == 3)
   
    /* input must be a string */
    if ( mxIsChar(prhs[0]) != 1)
      mexErrMsgIdAndTxt( "MATLAB:revord:inputNotString",
              "Input must be a string.");

    /* input must be a row vector */
    if (mxGetM(prhs[0])!=1)
      mexErrMsgIdAndTxt( "MATLAB:revord:inputNotVector",
              "Input must be a row vector.");
	char* input_buf = mxArrayToString(prhs[0]);
	mexPrintf("Load file %s\n", input_buf);
	PCloud pcloud;
	int np = pcloud.ReadFromPCD(input_buf);//np < 0 if error 
   
	mexPrintf("np: %d, dim: %d, tdim: %d, ", np, dim, tdim);
	mexPrintf("htype: %d, nn: %d, hs: %.2f, rho: %.2f, ", htype, nn, hs, rho);

	//cout<<"nn: "<<nn<<" hs: "<<hs<<" rho: "<<rho<<endl;
	
	double *II, *JJ, *SS, *points;
	points = mxGetPr(prhs[0]);

	vector<double> SSV;
	vector<unsigned int> IIV, JJV;

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

	mexPrintf("avers: %f, h: %f\n", avers, h);

	generate_laplace_matrix_sparse_matlab_dim2(pcloud, h, rho, IIV, JJV, SSV);
	
//	generate_pcdlaplace_matrix_sparse_matlab(points, np, dim, tdim, htype, nn, hs, rho, IIV, JJV, SSV);

	unsigned int nelem = IIV.size();
	if( nelem != JJV.size() ||  nelem != SSV.size() ){
		mexErrMsgTxt("Dimension of II, JJ, SS has to be the same");
	}
	plhs[0] = mxCreateDoubleMatrix(nelem, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(nelem, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(nelem, 1, mxREAL);
	II = mxGetPr(plhs[0]);
	JJ = mxGetPr(plhs[1]);
	SS = mxGetPr(plhs[2]);

	for(mwSize i = 0; i < nelem; i ++){
		II[i] = IIV[i];
		JJ[i] = JJV[i];
		SS[i] = SSV[i];
	}
}


