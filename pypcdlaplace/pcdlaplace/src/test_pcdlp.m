filepath = '/home/ming/training/mn2_c1/mn2_c100001.pcd';

opt.htype = 'psp';                                                                                  
opt.nn = 10;
opt.hs = 0.2;
opt.rho = 3;

[II, JJ, SS] = pcdlp(filepath, opt);