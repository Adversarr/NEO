function [M] = graphlp_matrix(X, tdim, opt)
%
% Compute the graph Laplace matrix from point cloud
%
% INPUTS
%  X: n-by-d data matrix, n: #points and d; ambient dimension.
%  tdim: the dimension of the manifold that points are sampled from.
%  opt.htype: the way to compute the parameter h. h = hs * neighborhoodsize
%             if htype = 'ddr' (data driven); h = hs if hytpe = 'psp' (pre-specify)
%             Default : 'ddr'
%  opt.nn:    the number of the nearest points that are used to estimated 
%             the neighborhood size.  
%             Default : 10, must > 1
%  opt.hs:    the scaling factor that scales the neighborhood size to the
%             parameter h	where h^2 = 4t.
%             Default: 2, must > 0
%  opt.rho:   The cut-off for Gaussion function evaluation. 
%             Default: 3, must > 0
%
% OUTPUTS
%  M: graph Laplace matrix
if nargin < 2
    error('Too few input arguments');	 
elseif nargin < 3
	opt.nn = 10;
	opt.hs = 2;
	opt.rho = 3;
	opt.htype = 'ddr';
end
opt=parse_opt(opt)

if opt.nn <= 1 | opt.hs <= 0 | opt.rho <= 0
	error('Invalid values in opt');
end

[II JJ SS] = graphlpmatrix(X, tdim, opt);
M=sparse(II, JJ, SS);

% Parsing Option.
function option = parse_opt(opt)
option = opt;
option_names = {'nn', 'hs', 'rho', 'htype'};
if ~isfield(option,'nn'),
	option = setfield(option,'nn', 10);
end
if ~isfield(option,'hs'),
	option = setfield(option,'hs',2);
end
if ~isfield(option,'rho'),
	option = setfield(option,'rho', 3);
end
if ~isfield(option,'htype'),
	option = setfield(option,'htype', 'ddr');
end


