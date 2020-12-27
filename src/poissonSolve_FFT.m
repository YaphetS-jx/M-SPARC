function [V] = poissonSolve_FFT(S,rhs)
% @brief    POISSONSOLVE solves the poisson equation for the 
%           electrostatic potential.
%
% @param poisson_tol    Tolerance for solving the poisson equation
%                       using iterative method AAR. 
% @param Isguess        1: guess vector provided,
%                       0: no guess vector available.
%
% @authors  Qimen Xu <qimenxu@gatech.edu>
%           Abhiraj Sharma <asharma424@gatech.edu>
%           Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
%
% @copyright (c) 2019 Material Physics & Mechanics Group, Georgia Tech
%

% t1 = tic;

if S.cell_typ < 3
	f = poisson_RHS(S,rhs);
else 
    error("not implemented.");
end

V = Pois_FFT_Periodic(S,f,S.Nx,S.Ny,S.Nz);
% fprintf(' Poisson problem solved by FFT took %fs\n',toc(t1));
end


function f = poisson_RHS(S,rhs)
% @brief	Poisson_RHS evaluates the right hand side of the poisson equation, 
%           including the boundary condtions, i.e. f = -4 * pi * ( rho + b - d) 
%           for cluster system, while for periodic system it's just 
%           f = -4 * pi * (rho + b).

f = -4 * pi * rhs;
% f = f - sum(f)./length(f);

if(S.BC ~= 2)
    error("Must use Piriodic B.C.");
end

end

function f = Pois_FFT_Periodic(S,rhs,N1,N2,N3)
rhs = reshape(rhs,N1,N2,N3);
g_hat = fftn(rhs);

f = ifftn(g_hat.*S.const_by_alpha);
f = f(:);
end
