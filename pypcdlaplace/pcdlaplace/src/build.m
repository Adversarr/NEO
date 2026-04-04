ipath =['-I' fullfile(matlabroot,'extern', 'include') ' -I/usr/include/CGAL'];

Lpath =['-L' fullfile(matlabroot,'bin','glnx64') ...
       ' -L' fullfile(matlabroot,'sys','os','glnx64')...
       ' -L/usr/lib'];
   %-lXmu -lXi -lXt -lXext -lICE
lpath =['-lmwlapack -lpthread -lCGAL -lgmp -lboost_thread'];

flag = '-largeArrayDims';
%mex('-v', ipath, 'graphlpmatrix.C', 'comp_llpmatrix.C', 'lpmatrix.C', 'point_cloud.C', Lpath, lpath);

%mex('-v', ipath, 'pcdlpmatrix.C', 'comp_llpmatrix.C', 'lpmatrix.C', 'point_cloud.C', Lpath, lpath);

mex('-v', ipath, 'pcdlp.cpp', 'comp_llpmatrix.cpp', 'point_cloud.cpp', Lpath, lpath);