function f = Poisson_RHS(rhs,S)
% @brief	Poisson_RHS evaluates the right hand side of the poisson equation, including
%			the boundary condtions, i.e. f = -4 * pi * ( rho + b - d) for cluster system,
%			while for periodic system it's just f = -4 * pi * (rho + b).
%
% @param rho		The electron density
% @param S			A struct that contains the following fields:
% @param S.b		The pseudocharge
% @param S.BC		Boundary condition: 1--isolated luster; 2--periodic system
% @param S.L1		The side of the domain in the x-direction
% @param S.L2		The side of the domain in the y-direction
% @param S.L3		The side of the domain in the z-direction
% @param S.Nx		The number of nodes in [0,L1]
% @param S.Ny		The number of nodes in [0,L2]
% @param S.Nz		The number of nodes in [0,L3]
% @param S.N		Total number of nodes
% @param S.FDn		Half of the order of the finite difference scheme
% @param S.w2		Finite difference weights of 2nd derivative
% @param S.W			Weights for spatial integration for Cuboidal domain
%
% @authors	Qimen Xu <qimenxu@gatech.edu>
%			Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
% @2016 Georgia Institute of Technology.

f = -4 * pi * (rhs);

if(S.BC == 1)
    % For cluster systems, we need to include boundary conditions d
    % RR = S.RR_AUG(S.isIn);
    for l = 0:S.l_cut
        multipole_moment(l+1).Qlm = sum(repmat(S.RR.^l .* (rhs) .* S.W,1,2*l+1).* S.SH(l+1).Ylm )';
    end
    
    % Calculate phi using multipole expansion
    phi = zeros(size(S.RR_AUG_3D));
    for l = 0 : S.l_cut
        denom = (2*l+1)*S.RR_AUG_3D.^(l+1);
        for m = -l : l
            Ylm_AUG_3D = reshape(S.SH(l+1).Ylm_AUG(:,m+l+1),size(phi));
            phi = phi + Ylm_AUG_3D .* multipole_moment(l+1).Qlm(m+l+1) ./ denom;
        end
    end
    phi = 4 * pi * phi;
    phi(S.isIn) = 0;
    
    % Calculate boundary conditions
    dx2 = S.dx * S.dx;
    dy2 = S.dy * S.dy;
    dz2 = S.dz * S.dz;
    d = zeros(S.Nx,S.Ny,S.Nz);
    II = (1+S.FDn):(S.Nx+S.FDn);
    JJ = (1+S.FDn):(S.Ny+S.FDn);
    KK = (1+S.FDn):(S.Nz+S.FDn);
    for p = 1:S.FDn
       d = d - S.w2(p+1)/dx2 * (phi(II+p,JJ,KK) + phi(II-p,JJ,KK));
       d = d - S.w2(p+1)/dy2 * (phi(II,JJ+p,KK) + phi(II,JJ-p,KK));
       d = d - S.w2(p+1)/dz2 * (phi(II,JJ,KK+p) + phi(II,JJ,KK-p));
    end
      
    d = reshape(d,[],1);
    
    f = f + d;
end