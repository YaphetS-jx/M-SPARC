function Vexx = evaluateExactExchangePotential(S,X,kptvec)
if S.ACEFlag == 0
    Vexx = zeros(S.N,size(X,2));
    V_guess = rand(S.N,1);
    for i = 1:size(X,2)
        for j = 1:S.Nev
            for q_ind = 1:S.tnkpthf
                % q_ind_rd is the index in reduced kptgrid
                q_ind_rd = S.kpthf_ind(q_ind,1);
                if S.kpthf_ind(q_ind,2)
                    psiq = S.psi_outer(:,j,q_ind_rd);
                else
                    psiq = conj(S.psi_outer(:,j,q_ind_rd));
                end

                rhs = conj(psiq).*X(:,i);
                if S.exxmethod == 0             % solving in fourier space
                    q = S.kptgridhf(q_ind,:);
                    k_shift = kptvec - q;
                    V_ji = poissonSolve_FFT(S,rhs,k_shift);
                else                            % solving in real space
                    f = poisson_RHS(S,rhs);
                    [V_ji, flag] = pcg(-S.Lap_std,-f,1e-8,1000,S.LapPreconL,S.LapPreconU,V_guess);
                    assert(flag==0);
                    V_guess = V_ji;
                end

                Vexx(:,i) = Vexx(:,i) - S.wkpthf(q_ind)*S.occ_outer(j,q_ind_rd)*V_ji.*psiq;
            end
        end
    end
else 
    if S.isgamma == 1
        Xi_times_psi = (transpose(S.Xi)*X)*S.dV;
        Vexx = -S.Xi*Xi_times_psi;
    else
        k_ind = find(ismembertol(S.kptgrid,kptvec,1e-8,'ByRows',true))+0;
        Xi_times_psi = S.Xi(:,:,k_ind)'*X*(S.dV);
        Vexx = - S.Xi(:,:,k_ind)*Xi_times_psi;
    end
end
    
end


function [V] = poissonSolve_FFT(S,rhs,k_shift)
shift_ind = find(ismembertol(S.k_shift,k_shift,1e-8,'ByRows',true))+0;
if shift_ind < S.num_shift
    u = rhs .* S.neg_phase(:,shift_ind);
else
    u = rhs;
end
u = reshape(u,S.Nx,S.Ny,S.Nz);
u_hat = fftn(u);
const_by_alpha = zeros(S.Nx,S.Ny,S.Nz);
const_by_alpha(:) = S.const_by_alpha(shift_ind,:,:,:);
V = ifftn(u_hat.*const_by_alpha);
if shift_ind < S.num_shift
    V = V(:) .* S.pos_phase(:,shift_ind);
else
    V = V(:);
end

if S.isgamma
    V = real(V(:));
end
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