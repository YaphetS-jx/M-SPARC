function [S] = evaluateExactExchangeEnergy(S)
S.Eex = 0;
% 
% w2 = S.w2;
N1 = S.Nx;
N2 = S.Ny;
N3 = S.Nz;
% V = S.L1*S.L2*S.L3;
% R_c = (3*V/(4*pi))^(1/3);
% 
% [I,J,K] = meshgrid(0:(N1-1),0:(N2-1),0:(N3-1));
% dx2 = S.dx*S.dx; dy2 = S.dy*S.dy; dz2 = S.dz*S.dz;
% 
% % alpha follows conjugate even space
% alpha = w2(1)*(1/dx2+1/dy2+1/dz2).*ones(N1,N2,N3);
% for k=1:S.FDn
%     alpha = alpha + w2(k+1)*2.*(cos(2*pi*I*k/N1)./dx2 + cos(2*pi*J*k/N2)./dy2 + cos(2*pi*K*k/N3)./dz2);
% end
% 
% alpha(1,1,1) = -2/R_c^2;
% 
% const = 1 - cos(R_c*sqrt(-1*alpha));
% const(1,1,1) = 1;
        
% V_guess = rand(S.N,1);
for i = 1:S.Nev
    for j = 1:S.Nev
        rhs = conj(S.psi_outer(:,i)).*S.psi_outer(:,j);
%         rhs = S.psi(:,i).*S.psi(:,j);
        
        
        % For periodic case
%         gij = poissonSolve_FFT(S,rhs);
        if S.BC == 2
            rhs = reshape(rhs,N1,N2,N3);
            ghat = fftn(rhs);
            
%             f = real(ifftn(-1*(ghat.*S.const_by_alpha)));
            f = ifftn(-1*(ghat.*S.const_by_alpha));

            f = f(:);
            rhs = rhs(:);
            
            S.Eex = S.Eex + 4*pi*S.occ_outer(i)*S.occ_outer(j)*sum(conj(rhs).*f.*S.W);
        end
        
        % for dirichlet case
        if S.BC == 1
            f = poisson_RHS(S,rhs);
            [gij, flag] = pcg(-S.Lap_std,-f,1e-8,1000,S.LapPreconL,S.LapPreconU,V_guess);
            assert(flag==0);
            V_guess = gij;

            S.Eex = S.Eex + S.occ_outer(i)*S.occ_outer(j)*sum(rhs.*gij.*S.W);
        end
    end
end

S.Etotal = S.Etotal + 0.25*S.Eex;
fprintf(' Eex = %.8f\n', S.Eex);
fprintf(' Etot = %.8f\n', S.Etotal);
fprintf(2,' ------------------\n');

fileID = fopen(S.outfname,'a');
fprintf(fileID,' Eex = %.8f\n', S.Eex);
fprintf(fileID,' Etot = %.8f\n', S.Etotal);
fclose(fileID);

end



function f = poisson_RHS(S,rhs)
if S.BC ~= 1
    error('Must be Dirichlet B.C.\n');
end

f = -4 * pi * (rhs);

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

dx2 = S.dx * S.dx;
dy2 = S.dy * S.dy;
dz2 = S.dz * S.dz;
d = zeros(S.Nx,S.Ny,S.Nz);

II = (1+S.FDn):(S.Nx+S.FDn);
JJ = (1+S.FDn):(S.Ny+S.FDn);
KK = (1+S.FDn):(S.Nz+S.FDn);

% only add charge correction on Dirichlet boundaries
for p = 1:S.FDn
    d = d - S.w2(p+1)/dx2 * (phi(II+p,JJ,KK) + phi(II-p,JJ,KK));
end
for p = 1:S.FDn
    d = d - S.w2(p+1)/dy2 * (phi(II,JJ+p,KK) + phi(II,JJ-p,KK));
end
for p = 1:S.FDn
    d = d - S.w2(p+1)/dz2 * (phi(II,JJ,KK+p) + phi(II,JJ,KK-p));
end

d = d(:);
f = f + d;
end