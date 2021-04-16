function [S] = evaluateExactExchangeEnergy(S)
S.Eex = 0;
if S.ACEFlag == 0
    V_guess = rand(S.N,1);
    for i = 1:S.Nev
        for j = 1:S.Nev
            for k_ind = 1:S.tnkpt
                for q_ind = 1:S.tnkpt
                    rhs = conj(S.psi_outer(:,i,q_ind)).*S.psi(:,j,k_ind);

                    if S.exxmethod == 0             % solving in fourier space
                        k = S.kptgrid(k_ind,:);
                        q = S.kptgrid(q_ind,:);
                        k_shift = k - q;
                        gij = poissonSolve_FFT(S,rhs,k_shift);
                    else                            % solving in real space
                        f = poisson_RHS(S,rhs);
                        [gij, flag] = pcg(-S.Lap_std,-f,1e-8,1000,S.LapPreconL,S.LapPreconU,V_guess);
                        assert(flag==0);
                        V_guess = gij;    
                    end

                    S.Eex = S.Eex + S.wkpt(k_ind)*S.wkpt(q_ind)*S.occ_outer(i)*S.occ_outer(j)*real(sum(conj(rhs).*gij.*S.W));
                end
            end
        end
    end
else
    psi_times_Xi = transpose(S.psi)*S.Xi;
    S.Eex = (transpose(S.occ_outer)*sum(psi_times_Xi.*psi_times_Xi,2))*(S.dx*S.dy*S.dz)^2;
end


S.Etotal = S.Etotal + S.hyb_mixing * S.Eex;
fprintf(' Eex = %.8f\n', S.Eex);
fprintf(' Etot = %.8f\n', S.Etotal);
fprintf(2,' ------------------\n');

fileID = fopen(S.outfname,'a');
fprintf(fileID,' Eex = %.8f\n', S.Eex);
fprintf(fileID,' Etot = %.8f\n', S.Etotal);
fclose(fileID);
end

function [V] = poissonSolve_FFT(S,rhs,k_shift)
sihft_ind = find(ismember(S.k_shift,k_shift,'rows'))+0;
% t1 = tic;
f = -4 * pi * rhs;
u = f .* exp(-1i*S.r*k_shift');
u = reshape(u,S.Nx,S.Ny,S.Nz);
u_hat = fftn(u);
const_by_alpha = zeros(S.Nx,S.Ny,S.Nz);
const_by_alpha(:) = S.const_by_alpha(sihft_ind,:,:,:);
V = ifftn(u_hat.*const_by_alpha);
V = V(:) .* exp(1i*S.r*k_shift');
if S.isgamma
    V = real(V(:));
end
% fprintf(' Poisson problem solved by FFT took %fs\n',toc(t1));
end



% copied from poissonSolve.m
function f = poisson_RHS(S,rhs)
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