function [S] = evaluateExactExchangeEnergy(S)
S.Eex = 0;

V_guess = rand(S.N,1);
for i = 1:S.Nev
    for j = 1:S.Nev
        rhs = conj(S.psi_outer(:,i)).*S.psi(:,j);

        % For periodic case
        if S.BC == 2
            gij = poissonSolve_FFT(S,rhs);
        end
        
        % for dirichlet case
        if S.BC == 1
            f = poisson_RHS(S,rhs);
            [gij, flag] = pcg(-S.Lap_std,-f,1e-8,1000,S.LapPreconL,S.LapPreconU,V_guess);
            assert(flag==0);
            V_guess = gij;    
        end
        
        S.Eex = S.Eex + S.occ_outer(i)*S.occ_outer(j)*sum(conj(rhs).*gij.*S.W);
    end
end

% S.Etotal = S.Etotal + 0.25*S.Eex;
S.Etotal = S.Etotal + S.Eex;
fprintf(' Eex = %.8f\n', S.Eex);
fprintf(' Etot = %.8f\n', S.Etotal);
fprintf(2,' ------------------\n');

fileID = fopen(S.outfname,'a');
fprintf(fileID,' Eex = %.8f\n', S.Eex);
fprintf(fileID,' Etot = %.8f\n', S.Etotal);
fclose(fileID);

end

function [V] = poissonSolve_FFT(S,rhs)
if(S.BC ~= 2)
    error("Must use Periodic BC\n");
end

% t1 = tic;
f = -4 * pi * rhs;
f = reshape(f,S.Nx,S.Ny,S.Nz);
g_hat = fftn(f);
V = ifftn(g_hat.*S.const_by_alpha);
V = V(:);
% fprintf(' Poisson problem solved by FFT took %fs\n',toc(t1));
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