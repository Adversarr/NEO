#ifndef __DATASTRUCTURE_H__
#define __DATASTRUCTURE_H__

///////////////////////////////////////////////////////
#include <CGAL/basic.h>
#include <CGAL/Cartesian.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Quotient.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/determinant.h>
#include <CGAL/iterator.h>
#include <CGAL/intersections.h>
#include <CGAL/enum.h>
#include <CGAL/Timer.h>
#include <CGAL/Origin.h>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <float.h>

#define MYNZERO 1e-10

//--------------------------------------------------
// #include <CGAL/Filtered_exact.h>
// typedef CGAL::Filtered_exact<double, CGAL::MP_Float>          NT;
// typedef CGAL::Cartesian<NT>                                   Rep;
//-------------------------------------------------- 

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel		 Rep;



/******************************************************************************/
/* 2D triangulation                                */    
/******************************************************************************/
#include <CGAL/Point_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Delaunay_triangulation_2.h>


typedef CGAL::Point_2<Rep>                         Point_2d;
typedef CGAL::Triangle_2<Rep>                                 Triangle_2d;


//------------------------------------------------------------------------------------
//Define a classes for the Regualar triangulation both 2D
// --------------------------------------------------------------
// Define a new Vertex_base with additional attributes.
// ----------------------- --------------------------------------
template < class Traits, class Vb = CGAL::Triangulation_vertex_base_2<Traits> >
class myVertex2D : public Vb {
public:
  typedef typename Vb::Point                  Point;
  typedef typename Vb::Face_handle        Face_handle;
  typedef typename Vb::Vertex_handle            Vertex_handle;
  
  typedef typename Point::R                          Rep;
  typedef CGAL::Vector_2<Rep>                        Vector;
  
  template < class TDS2 >
  struct Rebind_TDS {
     typedef typename Vb::template Rebind_TDS<TDS2>::Other  Vb2;
   typedef myVertex2D<Traits, Vb2>                        Other;
  };
  
  myVertex2D()                          {init();}
  myVertex2D( const Point& p) : Vb(p) {init();}
  myVertex2D( const Point& p, Face_handle cell) 
    : Vb(p, cell)                         {init();}
  myVertex2D( Face_handle cell) : Vb(cell)  {init();}
  
  int origin_id;    
  inline void init(){   origin_id = -1; }

  int get_origin_id(){return origin_id;}
  void set_origin_id(int id){ origin_id = id;}
};

//////////////////////////////////////////////////////////////////////////////
//Define a classes for the Regualar triangulation both 2D
// --------------------------------------------------------------------------
// Define a new Face_base                                                    
// --------------------------------------------------------------------------
template < class Traits, class Fb = CGAL::Triangulation_face_base_2<Traits> >
class myFace2D : public Fb 
{
public:
  typedef typename Traits::Point_2                      Point;
  typedef typename Fb::Vertex_handle                   Vertex_handle;
  typedef typename Fb::Face_handle                     Face_handle; 
 
  template < class TDS2 >
  struct Rebind_TDS {
  typedef typename Fb::template Rebind_TDS<TDS2>::Other  Fb2;
  typedef myFace2D<Traits, Fb2>                        Other;
  };
  
  myFace2D()                               { init(); }
  myFace2D( Vertex_handle v0, Vertex_handle v1, Vertex_handle v2) 
    : Fb(v0,v1,v2)                       { init(); }
  myFace2D( Vertex_handle v0, Vertex_handle v1, Vertex_handle v2, 
    Face_handle n0, Face_handle n1, Face_handle n2) 
    : Fb(v0,v1,v2,n0,n1,n2)           { init(); }
  
  inline void init(){  }
};

///////////////////////////////////////////////////////////////////////////
//data types
typedef Rep                                                   Triang_traits_2;
typedef myVertex2D<Triang_traits_2>                Vb2;
typedef myFace2D<Triang_traits_2>                              Fb2;
typedef CGAL::Triangulation_data_structure_2<Vb2, Fb2>        TDS_2;
typedef CGAL::Delaunay_triangulation_2<Triang_traits_2,TDS_2>     Triangulation_2;

typedef Triangulation_2::Vertex_handle              Vertex_handle_2d;
typedef Triangulation_2::Face_handle          Face_handle_2d;
typedef Triangulation_2::Face              Face_2d;
typedef Triangulation_2::Edge                       Edge_2d;


/******************************************************************************/
/* 3D triangulation                                */    
/******************************************************************************/

//for delaunay triangulation
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_3.h>
#include <CGAL/Triangulation_geom_traits_3.h>
#include <CGAL/Delaunay_triangulation_3.h>

typedef CGAL::Line_3<Rep>                                     Line_3d;
typedef CGAL::Point_3<Rep>                                    Point_3d;
typedef CGAL::Segment_3<Rep>                                  Segment_3d;
typedef CGAL::Ray_3<Rep>                                      Ray_3d;
typedef CGAL::Direction_3<Rep>                                Direction_3d;
typedef CGAL::Plane_3<Rep>                                    Plane_3d;
typedef CGAL::Triangle_3<Rep>                                 Triangle_3d;
typedef CGAL::Tetrahedron_3<Rep>                              Tetrahedron;
typedef CGAL::Vector_3<Rep>                                   Vector_3d;
typedef CGAL::Sphere_3<Rep>                                   Sphere_3d;


//////////////////////////////////////////////////////////////////////////////////////////////
//Define a classes for the delaunay triangulation in 3D
// --------------------------------------------------------------
// Define a new Vertex_base with additional attributes.
// ----------------------- --------------------------------------

#define VERTEX_FLAG_VISITED    0x01     //the vertex has been visited
#define VERTEX_FLAG_IN      0x02     //the vertex is in 
template < class Traits, class Vb = CGAL::Triangulation_vertex_base_3<Traits> >
class myVertex3D : public Vb {
public:
  typedef typename Vb::Point                                  Point;
  typedef typename Vb::Cell_handle                            Cell_handle;
  typedef typename Vb::Vertex_handle                          Vertex_handle;
  
//  typedef typename Point::R                          Rep;
//  typedef CGAL::Vector_3<Rep>                        Vector;
  
  template < class TDS2 >
  struct Rebind_TDS {
     typedef typename Vb::template Rebind_TDS<TDS2>::Other    Vb2;
   typedef myVertex3D<Traits, Vb2>                            Other;
  };
  
  myVertex3D()                {init();}
  myVertex3D( const Point& p) : Vb(p) {init();}
  myVertex3D( const Point& p, Cell_handle cell) 
    : Vb(p, cell)                         {init();}
  myVertex3D( Cell_handle cell) : Vb(cell)  {init();}
  
  //if the vertex from the original surface, origin_id is its index in the original surface
  //otherwise -1;
  int origin_id; 
  unsigned char flag;  
  
//  Point v_pole;
  inline void init(){
    origin_id = -1;
    flag = 0;
//    v_pole = this->point();
  }
  
  bool check_flag(unsigned char f){ return flag&f; }
  void set_flag(unsigned char f){ flag |= f; }
  void un_set_flag(unsigned char f){ flag &= (~f); }
    
  int get_origin_id(){return origin_id;}
  void set_origin_id(int id){origin_id = id;}

};


// --------------------------------------------------------------------------
// Define a new Cell_base                                                    
// --------------------------------------------------------------------------

//facet flag
#define FACET_FLAG_VISITED        0x1    
#define FACET_FLAG_DEEPINT        0x2  //the cell deeply intersects with the adjacent cell along the ith facet
#define FACET_FLAG_INTERSECT      0x4  //the dual voronoi edge intersects mesh

//cell flag
#define CELL_FLAG_VISITED         0x01 //mark the cell as visited
#define CELL_FLAG_IN              0x02 //mark the cell when it is in 

#define CELL_FLAG_INVALID_CC      0x04 //the circumcenter of the cell can not be calculated properly 
#define CELL_FLAG_HAVING_CC       0x08 //the circumcenter of the cell has been calculated

#define CELL_FLAG_INSIDE          0x10 //mark the cell which is inside the given mesh
#define CELL_FLAG_OUTSIDE         0x20 //mark the cell which is inside the given mesh
#define CELL_FLAG_TINSIDE          0x10 //temporarily mark the cell which is inside the given mesh
#define CELL_FLAG_TOUTSIDE         0x20 //temporarily mark the cell which is inside the given mesh


#define CELL_FLAG_BBALL           0x40 //mark the cell whics is big
template < class Traits, class Cb = CGAL::Triangulation_cell_base_3<Traits> >
class myCell3D : public Cb {
public:
  typedef typename Traits::Point_3                            Point;
  typedef typename Cb::Vertex_handle                          Vertex_handle;
  typedef typename Cb::Cell_handle                            Cell_handle; 
 
  template < class TDS2 >
  struct Rebind_TDS {
  typedef typename Cb::template Rebind_TDS<TDS2>::Other       Cb2;
  typedef myCell3D<Traits, Cb2>                               Other;
  };
  
  myCell3D()                               { init(); }
  myCell3D( Vertex_handle v0, Vertex_handle v1, Vertex_handle v2, Vertex_handle v3) 
    : Cb(v0,v1,v2,v3)                       { init(); }
  myCell3D( Vertex_handle v0, Vertex_handle v1, Vertex_handle v2, Vertex_handle v3,
    Cell_handle n0, Cell_handle n1, Cell_handle n2, Cell_handle n3) 
    : Cb(v0,v1,v2,v3,n0,n1,n2,n3)           { init(); }
 
 
  int cell_id; 
  int facet_id[4];       
  unsigned short cell_flag;
  unsigned char facet_flag[4];
  
  Point voronoi_vertex;  
  
  inline void init(){
    for (unsigned i = 0 ; i < 4 ; i++) {
    facet_id[i] = -1;
    facet_flag[i] = 0;
  }
  cell_id = -1;
  cell_flag = 0;
  voronoi_vertex = CGAL::ORIGIN;
  }

  bool check_cell_flag(unsigned short f){ return cell_flag&f; }
  void set_cell_flag(unsigned short f){ cell_flag |= f; }
  void un_set_cell_flag(unsigned short f){ cell_flag &= (~f); }
  
  bool check_ith_facet_flag(int i, unsigned char f){ return facet_flag[i]&f; }
  void set_ith_facet_flag(int i, unsigned char f){ facet_flag[i] |= f; }
  void un_set_ith_facet_flag(int i, unsigned char f){ facet_flag[i] &= (~f); }
    
  int get_cell_id(){return cell_id;}
  void set_cell_id(int id){cell_id = id;}
  int get_facet_id(int i){return facet_id[i];}
  void set_cell_id(int i, int id){facet_id[i] = id;}

  void  set_voronoi( const Point &p)    { voronoi_vertex = p; }
  Point get_voronoi() const { return voronoi_vertex; }

//  double cell_sq_radius(){ 
//  return CGAL::to_double((voronoi_vertex - vertex(0)->point()) * (voronoi_vertex - vertex(0)->point()) ); 
//  }

};



///////////////////////////////////////////////////////////////////////////
//data types for delaunay
typedef CGAL::Triangulation_geom_traits_3<Rep>                Triang_traits;
typedef myVertex3D<Triang_traits>                Vb3;
typedef myCell3D<Triang_traits>                              Cb3;
typedef CGAL::Triangulation_data_structure_3<Vb3, Cb3>        TDS;
typedef CGAL::Delaunay_triangulation_3<Triang_traits, TDS>     Triangulation;


typedef Triangulation::Vertex            Vertex_3d;
typedef Triangulation::Vertex_handle        Vertex_handle_3d;
typedef Triangulation::Cell              Cell_3d;
typedef Triangulation::Cell_handle          Cell_handle_3d;
typedef Triangulation::Edge              Edge_3d;
typedef Triangulation::Facet            Facet_3d;



//====================================================================
//Begin: Define classes for the dD spatial searching
//====================================================================
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_d.h>
#include <CGAL/Cartesian_d.h>  
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Delaunay_d.h>

typedef CGAL::Cartesian_d<double> KCd;
typedef CGAL::Point_d<KCd> dPoint;         
typedef CGAL::Vector_d<KCd> dVector;
//typedef K::Point_d dPoint;
//typedef K::Vector_d dVector;
typedef KCd::FT FT;

typedef CGAL::Delaunay_d<KCd> Triangulation_d;
typedef Triangulation_d::Point_d Point_d;
typedef Triangulation_d::Simplex_handle dSimplex_handle;
typedef Triangulation_d::Vertex_handle dVertex_handle;


//====================================================================
//End: Base elements for dD spatial search
//====================================================================

/*
class dPoint{
public:
  //Constructor
  	dPoint(): _flag(0){}
  	dPoint(const vector<double>& co): _flag(0) {  _coord = co; }
  
  	dPoint(const dPoint& dP){
    	_coord = dP.coord(); 
   	_flag = dP.flag();
  	}
  	//Overloaded operator
  	dPoint& operator = (const dPoint& dP){
    	if (this != &dP){
			this->dPoint::~dPoint();
			new (this) dPoint(dP);
		}
		return *this;
  	}
	//Function for accessing data
  	const vector<double>& coord() const { return _coord; }
 	void set_coord(const vector<double>& co) { _coord = co; }

  	bool check_flag(unsigned char f) { return _flag&f; }
  	void set_flag(unsigned char f) { _flag |= f; }
  	void un_set_flag(unsigned char f) { _flag &= (~f); }
  	unsigned char flag() const { return _flag; }
  	void copy_flag(unsigned char f) { _flag = f; }
  
private:
  	vector<double> _coord;  
  	unsigned char _flag;
};
*/

//====================================================================
//End: Base elements for dD spatial search
//====================================================================

struct ProjPoint{
	ProjPoint():vid(-1), area(-1), sqdist(-1){}
	ProjPoint(int v, double a, double d): vid(v), area(a), sqdist(d){}
	int vid;
	double area;
	double sqdist;
}; 

enum pmode{
	EIGEN, FUNCTION
};
enum mmode{
	GRAPH, PCLOUD	
};

enum nmode{
	DENSE, SPARSE	
};

enum cmode{
	MAT, CXX 
};

enum stype{
	SYM, NONSYM, SPHERE, PLANE, SKINSURF 
};


#endif //__DATASTRUCTURE_H__
