function S = ace_operator(S)
% Ns = S.Nev;
Ns = sum(S.occ_outer>1e-6);

S.Xi = zeros(S.N,Ns);    % For storage of W and Xi
V_guess = rand(S.N,1);
rhs = zeros(S.N,Ns);

for i = 1:Ns
    rhs(:,i:Ns) = bsxfun(@times,S.psi_outer(:,i:Ns),S.psi_outer(:,i));
    V_i = zeros(S.N,Ns);
    for j = i:Ns
        if (S.occ_outer(i) + S.occ_outer(j) > 1e-4)
            if S.exxmethod == 0             % solving in fourier space
                V_i(:,j) = poissonSolve_FFT(S,rhs(:,j));
            else                            % solving in real space
                f = poisson_RHS(S,rhs(:,j));
                [V_i(:,j), flag] = pcg(-S.Lap_std,-f,1e-8,1000,S.LapPreconL,S.LapPreconU,V_guess);
                assert(flag==0);
                V_guess = V_i(:,j);
            end
        end
    end
    S.Xi(:,(i+1:Ns)) = S.Xi(:,(i+1:Ns)) - S.occ_outer(i)*bsxfun(@times,S.psi_outer(:,i),V_i(:,(i+1:Ns)));
    S.Xi(:,i) = S.Xi(:,i) - bsxfun(@times,S.psi_outer(:,(i:Ns)),V_i(:,(i:Ns))) * S.occ_outer((i:Ns));
end

M = (transpose(S.psi_outer(:,1:Ns))*S.Xi)*S.dx*S.dy*S.dz;
L = chol(-M); 
S.Xi = S.Xi * inv(L); % Do it efficiently

end


function [V] = poissonSolve_FFT(S,rhs)
% t1 = tic;
f = -4 * pi * rhs;
f = reshape(f,S.Nx,S.Ny,S.Nz);
g_hat = fftn(f);
V = ifftn(g_hat.*S.const_by_alpha);
V = real(V(:));
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
