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

%[S.phi,conv_flag, relres, iter] = gmres(S.Lap_std,f,50,poisson_tol,50,S.LapPreconL,S.LapPreconU,phi_guess);
% S.phi = aar(S.Lap_std,f,phi_guess,poisson_tol,S.MAXIT_POISSON,0.6,0.6,7,6,S.LapPreconL,S.LapPreconU);

V = Pois_FFT_Periodic(f,S.w2,S.FDn,S.Nx,S.Ny,S.Nz,S.dx,S.dy,S.dz);

% assert(norm(S.Lap_new(V)-f)<1e-8);
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

function f = Pois_FFT_Periodic(rhs,w2,FDn,N1,N2,N3,dx,dy,dz)
rhs = reshape(rhs,N1,N2,N3);

g_hat = fftn(rhs);

[I,J,K] = meshgrid(0:(N1-1),0:(N2-1),0:(N3-1));

dx2 = dx*dx; dy2 = dy*dy; dz2 = dz*dz;

% alpha follows conjugate even space
alpha = w2(1)*(1/dx2+1/dy2+1/dz2).*ones(N1,N2,N3);
for k=1:FDn
    alpha = alpha + w2(k+1)*2.*(cos(2*pi*I*k/N1)./dx2 + cos(2*pi*J*k/N2)./dy2 + cos(2*pi*K*k/N3)./dz2);
end

alpha(1,1,1) = 1;

f = real(ifftn(g_hat./alpha));
f = f(:);
end
