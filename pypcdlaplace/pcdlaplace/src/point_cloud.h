#ifndef __POINT_CLOUD_H__
#define __POINT_CLOUD_H__

#include "datastructure.h"
#include <vector>
using namespace std;

//-------------------------------------------------------------------
//VPCloud: vertex class for triangle mesh
//-------------------
#define VPCLOUD_FLAG_SELECTED	0x4


class VPCloud{

public:
  //Constructor
  	VPCloud(): _flag(0){}
  	explicit VPCloud(const dPoint& co): _flag(0), _coord(co) {}
  	VPCloud(const VPCloud& dP) = default;
  	VPCloud& operator = (const VPCloud& dP) = default;
	//Function for accessing data
  	const dPoint& coord() const { return _coord; }
  	dPoint& coord() { return _coord; }
 	void set_coord(const dPoint& co) { _coord = co; }

  	bool check_flag(unsigned char f) const { return _flag&f; }
  	void set_flag(unsigned char f) { _flag |= f; }
  	void un_set_flag(unsigned char f) { _flag &= (~f); }
  	unsigned char flag() const { return _flag; }
  	void copy_flag(unsigned char f) { _flag = f; }
  
private:
  	unsigned char _flag;
  	dPoint _coord;  
};

//-------------------------------------------------------------------


class PCloud{
public:
  	//Constructor
  	PCloud():_d(0), num_points(0){}
	PCloud(const double *points, unsigned int np, unsigned int dim);
  	//Destructor
	~PCloud() = default;

  	//Function for accessing data
    unsigned int p_count() const { return num_points; }

  	VPCloud& point(unsigned int i ) { return _points[i]; }
  	const VPCloud& point(unsigned int i ) const { return _points[i]; }
		
  	void copy_name(char* name) const { sprintf(name, "%s", _name); }	
  	void set_name(const char* name){ sprintf(_name, "%s", name );}
	
	unsigned int dd() const { return _d; }
	void set_dd(unsigned int d) { _d = d; }


	//Functions for output 
  	//void OutPCloudOffFile(char *filename);
  	//void OutPCloudOffFile(FILE *fp, double r, double g, double b, double a);
	void OutPCloud(char *filename);


  	//Functions for rendering
  	//void Render(float m[16]);
  	//void Render(float m[16], vector<double> fn, double min, double max, bool shownormal);
	//void Render_select_points(vector<unsigned int>& select_points);
	//void SelectRender_points();

  	//Functions for bounding box
  	void GetBBox();
  	dPoint pmin(){ return _pmin; }
  	dPoint pmax(){ return _pmax; }
  	double radius() {return sqrt( CGAL::to_double((_pmax - _pmin) * (_pmax - _pmin)) ) / 2; }

	//Average neighborhood size
  	double average_size(unsigned int k);
  	//Functions for Creation
 // 	bool ReadPointCloud(char *filename);
  	int ReadFromPCD(char *filename);

  	//Clear
//  	void clear(){ _points.clear();  }
	void clear(){ _points.clear(); num_points = 0; }
            
private:
  unsigned int _d;
  vector<VPCloud> _points;
  unsigned int num_points;
  dPoint _pmin;
  dPoint _pmax;
  char _name[256];
};

#endif //__POINT_CLOUD_H__
