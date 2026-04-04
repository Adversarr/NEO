#include <math.h>
#include <queue>
#include <fstream>
#include <stdexcept>

#include "point_cloud.h"

double __partcolorgrtable[31][3] =
  {
    { 0, 0, 1 },
    { 0, 0.0625, 0.9375 },
    { 0, 0.1250, 0.8750 },
    { 0, 0.1875, 0.8125 },
    { 0, 0.2500, 0.7500 },
    { 0, 0.3125, 0.6875 },
    { 0, 0.3750, 0.6250 },
    { 0, 0.4375, 0.5625 },
    { 0, 0.5625, 0.4375 },
    { 0, 0.6250, 0.3750 },
    { 0, 0.6875, 0.3125 },
    { 0, 0.7500, 0.2500 },
    { 0, 0.8125, 0.1875 },
    { 0, 0.8750, 0.1250 },
    { 0, 0.9375, 0.0625 },

    { 0, 1, 0 },
    { 0.0625, 0.9375, 0 },
    { 0.1250, 0.8750, 0 },
    { 0.1875, 0.8125, 0 },
    { 0.2500, 0.7500, 0 },
    { 0.3125, 0.6875, 0 },
    { 0.3750, 0.6250, 0 },
    { 0.4375, 0.5625, 0 },
    { 0.5625, 0.4375, 0 },
    { 0.6250, 0.3750, 0 },
    { 0.6875, 0.3125, 0 },
    { 0.7500, 0.2500, 0 },
    { 0.8125, 0.1875, 0 },
    { 0.8750, 0.1250, 0 },
    { 0.9375, 0.0625, 0 },
    { 1, 0, 0 } };

void
get_color(double v, double minv, double maxv, double c[3])
{
  double scale = 0.8;
  /*
   //---------------------------------------------------------------------------------------
   //all warm to cold
   double nnv = maxv - minv;
   int minc = 0, maxc = 63;
   double vs = (v - minv) / nnv * (maxc - minc + 1);

   int i = (int)(vs);
   for(int j = 0; j < 3; j ++)
   c[j] = scale * ((vs - i) * (__colortable[i + 1][j] - __colortable[i][j]) + __colortable[i][j]);

   //  cout<<"c: "<<c[0]<<" "<<c[1]<<" "<<c[2]<<endl;
   //---------------------------------------------------------------------------------------
   */

  //maxv = 2.5;
  //minv = 1;
  double nnv = maxv - minv;

  if (fabs(nnv) < MYNZERO)
    {
      for (int j = 0; j < 3; j++)
        c[j] = scale * __partcolorgrtable[15][j];
      return;
    }

  //---------------------------------------------------------------------------------------
  //warm to cold
  int minc = 0, maxc = 30;
  if (v <= minv)
    {
      for (int j = 0; j < 3; j++)
        c[j] = scale * __partcolorgrtable[minc][j];
      return;
    }

  if (v >= maxv)
    {
      for (int j = 0; j < 3; j++)
        c[j] = scale * __partcolorgrtable[maxc][j];
      return;
    }
  double vs = (v - minv) / nnv * (maxc - minc);

  int i = (int) (vs);
  for (int j = 0; j < 3; j++)
    c[j] = scale
        * ((vs - i) * (__partcolorgrtable[i + 1][j] - __partcolorgrtable[i][j])
            + __partcolorgrtable[i][j]);
  //  cout<<"c: "<<c[0]<<" "<<c[1]<<" "<<c[2]<<endl;
  //---------------------------------------------------------------------------------------

  /*
   //---------------------------------------------------------------------------------------
   //grey scale

   double maxc = 0, minc = 0.5;

   if(v <= minv) { c[0] = c[1] = c[2] = minc; return; }
   if(v >= maxv) { c[0] = c[1] = c[2] = maxc; return; }

   double t = (v - 1) / nnv;
   c[0] = c[1] = c[2] = minc  + (maxc - minc) * t;

   //---------------------------------------------------------------------------------------
   */
}

PCloud::PCloud(const double *points, unsigned int np, unsigned int dim)
  : _d(dim), _points(np), num_points(np)
{
  vector<double> coords;
  coords.resize(dim);
  if (points == NULL && np > 0)
    {
      throw std::invalid_argument("Point buffer must not be null");
    }

  for (unsigned int i = 0; i < np; i++)
    {
      for (unsigned int j = 0; j < dim; j++)
        {
          coords[j] = points[i * dim + j];
        }
      _points[i].set_coord(dPoint(dim, coords.begin(), coords.end()));
    }
}
/*
 bool PCloud::ReadPointCloud(char *filename)
 {
 ifstream fin;
 fin.open(filename);
 if(!fin){
 cout<<"Cannot open the file "<<filename<<endl;
 return false;
 }

 istream_iterator<double> input(fin);
 istream_iterator<double> end;

 if(input == end){
 cout<<"Empty data "<<endl;
 return false;
 }

 unsigned int d = (unsigned int)(*input); ++input;
 set_dd(d);
 cout<<"d: "<<dd()<<endl;

 vector<double> coord;
 coord.resize(d);
 while ( input != end) {
 for(unsigned int i = 0; i < d - 1; i ++){
 coord[i] = *input; ++input;
 //cout<<coord[i]<<" ";

 if( input == end ){
 cout<<"Invalid data"<<endl;
 return false;
 }
 }
 coord[d - 1] = *input; ++input;
 //cout<<coord[d - 1]<<endl;
 add_point( VPCloud( dPoint(d, coord.begin(), coord.end()) ) );
 }
 fin.close();
 //GetBBox();
 return true;
 }
 */
int PCloud::ReadFromPCD(char *filename)
{
  _d = 3;
  using namespace std;
  ifstream fin;
  fin.open(filename);
  if (!fin)
    {
      cout << "Cannot open the file " << filename << endl;
      return -1;
    }
#define MAX_CHARS_PER_LINE 256
  char buf[MAX_CHARS_PER_LINE];
  char token[32];
//	int num_point;

  // read the header
  while (!fin.eof())
    {
      // read an entire line into memory
      fin.getline(buf, MAX_CHARS_PER_LINE);

      sscanf(buf, "%s %*s", token);
      if (strcmp(token, "POINTS") == 0)
        {
          sscanf(buf, "%s %d %*s", token, &num_points);
        }
      else if (strcmp(token, "DATA") == 0)
        {
          sscanf(buf, "%*s %s", token);
          if (strcmp(token, "ASCII") != 0 && strcmp(token, "ascii") != 0)
            {
              //		  cerr<<"Error, cannot read other than ASCII type"<<endl;
              return -2;
            }
          break;
        }
    }

  vector<double> coord(3, 0.0);
  int i = 0;
  clear();
  _points.resize(num_points);
  /*
   //	mexPrintf("Size of _points = %d, capacity =%d\n", (int)_points.size(), (int)_points.capacity());
   for(i=0; i< num_points; ++i)
   {
   //		add_point( VPCloud( dPoint(3, coord.begin(), coord.end()) ) );
   dPoint p(3, coord.begin(), coord.end());
   pcl[i].set_coord(p);
   mexPrintf("%lf %lf %lf %d\n", p[0], p[1], p[2], i);
   }

   mexPrintf("Processed\n");
   //	_points.assign(pcl, pcl+num_point);
   */

  fin.getline(buf, MAX_CHARS_PER_LINE);
  while (!fin.eof())
    {
      sscanf(buf, "%lf %lf %lf", &coord[0], &coord[1], &coord[2]);
      dPoint p(3, coord.begin(), coord.end());
      _points[i].set_coord(p);
      ++i;
//		mexPrintf("%lf %lf %lf %d\n", p[0], p[1], p[2], i);
//		add_point( VPCloud( dPoint(3, coord.begin(), coord.end()) ) );
      fin.getline(buf, MAX_CHARS_PER_LINE);
    }

  fin.close();
  return num_points;
}

////////////////////////////////////////////////////////////////////////////////////////
//GetBBox
//--------
//Get a box which is twice as bigger as the boundding box of the  object
//
void
PCloud::GetBBox()
{
  typedef KCd::FT FT;

  if (p_count() == 0)
    {
      return;
    }
  vector<FT> min;
  min.resize(dd(), FT(FLT_MAX));
  vector<FT> max;
  max.resize(dd(), FT(-FLT_MAX));

  for (unsigned int i = 0; i < p_count(); i++)
    {
      for (unsigned int j = 0; j < dd(); j++)
        {
          if ((point(i).coord())[j] < min[j])
            {
              min[j] = (point(i).coord())[j];
            }
          else if ((point(i).coord())[j] > max[j])
            {
              max[j] = (point(i).coord())[j];
            }
        } //for j
    } //for i

  _pmin = dPoint(dd(), min.begin(), min.end());
  _pmax = dPoint(dd(), max.begin(), max.end());

  cerr << "BBox: min " << _pmin << " max " << _pmax << endl;
}

double
PCloud::average_size(unsigned int k)
{
  typedef CGAL::Search_traits_d<KCd> Traits;
  typedef CGAL::Euclidean_distance<Traits> Distance;
  typedef CGAL::Fair<Traits> Fair;
  typedef CGAL::Orthogonal_k_neighbor_search<Traits, Distance, Fair> Neighbor_search;
  typedef Neighbor_search::Tree Tree;
  typedef KCd::FT FT;

  unsigned int np = p_count();
  if (np == 0)
    {
      return 0.0;
    }

  vector<dPoint> points;
  points.reserve(np);
  for (unsigned int i = 0; i < np; i++)
    {
      points.push_back(point(i).coord());
    }

  Fair fair(10);
  Tree tree(points.begin(), points.end(), fair);

  double eps = 1e-3;
  double as = 0;
  for (unsigned int i = 0; i < np; i++)
    {
      dPoint pt = point(i).coord();
      Neighbor_search search(tree, pt, k, FT(eps));

      FT dist(0);
      unsigned int nfound = 0;
      for (Neighbor_search::iterator iter = search.begin();
          iter != search.end(); iter++)
        {
          dist += sqrt(iter->second);
          nfound++;
        }
      if (nfound > 0)
        {
          as += (dist / nfound);
        }
    }
  return as / np;
}

void
PCloud::OutPCloud(char *filename)
{
  ofstream fout;
  fout.open(filename);
  if (fout.fail())
    {
      std::cout << "Failed to open file!" << std::endl;
      return;
    }
  for (unsigned int i = 0; i < p_count(); i++)
    {
      fout << point(i).coord() << endl;
    }
  fout.close();
}

/*
 //================================================================
 //begin: off file outpur function
 //================================================================
 void PCloud::OutPCloudOffFile(char *filename)
 {
 FILE *fp;
 if( (fp = fopen(filename, "w")) == NULL ){
 std::cout<<"Failed to open file!"<<std::endl;
 return;
 }
 fprintf(fp, "LIST\n");
 fprintf(fp, "appearance {linewidth 4}\n");
 fprintf(fp, "{\n");
 fprintf(fp, "OFF\n");

 fprintf(fp, "%d %d 0\n", p_count(), p_count());
 for(unsigned int i = 0; i < p_count(); i ++){
 fprintf(fp, "%f %f %f\n", (point(i).coord())(0), (point(i).coord())(1), (point(i).coord())(2));
 }
 for(unsigned int i = 0; i < p_count(); i ++){
 if(  point(i).check_flag(VPCLOUD_FLAG_SELECTED) ){
 fprintf(fp, "1 %d 0 0 0 1\n", i);
 }
 else{
 fprintf(fp, "1 %d 1 1 1 1\n", i);
 }
 }
 fprintf(fp, "}\n");

 fclose(fp);
 }

 void PCloud::OutPCloudOffFile(FILE *fp, double r, double g, double b, double a)
 {
 if( fp == NULL ){
 std::cout<<"Invalid FILE pointer"<<std::endl;
 return;
 }

 fprintf(fp, "{\n");
 fprintf(fp, "OFF\n");
 fprintf(fp, "%d %d 0\n", p_count(), p_count());
 for(unsigned int i = 0; i < p_count(); i ++){
 fprintf(fp, "%f %f %f\n", (point(i).coord())(0), (point(i).coord())(1), (point(i).coord())(2));
 }
 for(unsigned int i = 0; i < p_count(); i ++){
 if( point(i).check_flag(VPCLOUD_FLAG_SELECTED) )
 fprintf(fp, "1 %d 1 0 0 1\n", i);
 else
 fprintf(fp, "1 %d %f %f %f %f\n", i, r, g, b, a);
 }
 fprintf(fp, "}\n");
 }
 */
/*
 void PCloud::Render(unsigned int indx, unsigned int indy, unsigned int indz)
 {
 glDisable(GL_LIGHTING);
 //glColor4f(1.0, 0, 0, 0.5);
 glPointSize(4);

 for(unsigned int i = 0; i < p_count(); i ++){
 glBegin(GL_POINTS);

 if(point(i).check_flag(VPCLOUD_FLAG_SELECTED)){
 glColor4f(1, 0, 0, 1);
 }
 else{
 glColor4f(0, 0, 0, 1);
 }
 glVertex3f( (point(i).coord())(0),
 (point(i).coord())(1),
 (point(i).coord())(2));
 glEnd();
 }

 glPointSize(1);
 glEnable(GL_LIGHTING);

 double scale = 0.2;
 for(unsigned int i = 0; i < p_count(); i ++){
 glBegin(GL_LINES);

 VECTOR3 vv = point(i).coord() + scale * point(i).normal();
 glVertex3f( (point(i).coord())(0),
 (point(i).coord())(1),
 (point(i).coord())(2));

 glVertex3f( vv(0), vv(1), vv(2));

 glEnd();
 }
 }

 void PCloud::Render(, vector<double> fn, double min, double max, bool shownormal)
 {
 glDisable(GL_LIGHTING);
 //glColor4f(1.0, 0, 0, 0.5);

 double xoffset = 0;
 double c[3];
 GLfloat mat_diffuse[4];
 GLfloat mat_ambient[4]={0.2, 0.2, 0.2, 1.0};

 glPointSize(8);

 for(unsigned int i = 0; i < p_count(); i ++){

 glBegin(GL_POINTS);

 get_color(fn[i], min, max, c);
 glColor4f(c[0], c[1], c[2], 1.0f);
 glVertex3f( (point(i).coord())(0) - xoffset ,
 (point(i).coord())(1),
 (point(i).coord())(2));
 glEnd();
 }

 glPointSize(1);

 glEnable(GL_LIGHTING);
 glLineWidth(3);
 if(shownormal){
 double scale = 0.4;
 for(unsigned int i = 0; i < p_count(); i ++){
 glBegin(GL_LINES);
 VECTOR3 vv = point(i).coord() + scale * point(i).normal();
 glVertex3f( (point(i).coord())(0),
 (point(i).coord())(1),
 (point(i).coord())(2));
 glVertex3f( vv(0), vv(1), vv(2));
 glEnd();
 }
 }
 glLineWidth(1);

 }
 */
/*
 void PCloud::Render_select_vertices(vector<unsigned int>& select_vertices)
 {
 glDisable(GL_LIGHTING);
 glColor4f(0.0, 0.0, 0, 1.0);
 glPointSize(6);

 glBegin(GL_POINTS);

 for(vector<unsigned int>::iterator vit = select_vertices.begin(); vit != select_vertices.end(); vit ++){
 glVertex3f( (point(*vit).coord())(0), (point(*vit).coord())(1), (point(*vit).coord())(2) );
 }
 glEnd();
 glPointSize(1);
 glEnable(GL_LIGHTING);
 }

 void PCloud::SelectRender_vertices()
 {
 glInitNames();
 glPushName(0);
 for(unsigned int i = 0; i < p_count(); i ++){
 glLoadName(i);
 glBegin(GL_POINTS);
 glVertex3f( (point(i).coord())(0), (point(i).coord())(1), (point(i).coord())(2));
 glEnd();
 }
 }


 //==============================================================
 //end: render functions
 //===============================================================
 */
