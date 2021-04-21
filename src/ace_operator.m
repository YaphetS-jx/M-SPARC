function S = ace_operator(S)
if S.exxmethod == 1
    V_guess = rand(S.N,1);
end

if S.isgamma == 1
    Ns = sum(S.occ_outer>1e-6);
    S.Xi = zeros(S.N,Ns);    % For storage of W and Xi
    rhs = zeros(S.N,Ns);
    for i = 1:Ns
        rhs(:,i:Ns) = bsxfun(@times,S.psi_outer(:,i:Ns),S.psi_outer(:,i));
        V_i = zeros(S.N,Ns);
        for j = i:Ns
            if (S.occ_outer(i) + S.occ_outer(j) > 1e-4)
                if S.exxmethod == 0             % solving in fourier space
                    V_i(:,j) = poissonSolve_FFT(S,rhs(:,j),[0,0,0]);
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

else
    Ns = S.Nev;
    S.Xi = zeros(S.N,Ns,S.tnkpt);    % For storage of W and Xi
    for k_ind = 1:S.tnkpt
        for q_ind = 1:S.tnkpthf
            % q_ind_rd is the index in reduced kptgrid
            q_ind_rd = S.kpthf_ind(q_ind,1);
            
            k = S.kptgrid(k_ind,:);
            
            if S.kpthf_ind(q_ind,2)
                psi_q_set = S.psi_outer(:,1:Ns,q_ind_rd);
                q = S.kptgrid(q_ind_rd,:);
            else
                psi_q_set = conj(S.psi_outer(:,1:Ns,q_ind_rd));
                q = -S.kptgrid(q_ind_rd,:);
            end
            
            for i = 1:Ns
                psi_k = S.psi_outer(:,i,k_ind);
                rhs = conj(psi_q_set) .* psi_k;
                k_shift = k - q;
                V_i = zeros(S.N,Ns);
                for j = 1:Ns
                    if S.occ_outer(j,q_ind_rd) > 1e-6
                        if S.exxmethod == 0             % solving in fourier space
                            V_i(:,j) = poissonSolve_FFT(S,rhs(:,j),k_shift);
                        else                            % solving in real space
                            f = poisson_RHS(S,rhs(:,j));
                            [V_i(:,j), flag] = pcg(-S.Lap_std,-f,1e-8,1000,S.LapPreconL,S.LapPreconU,V_guess);
                            assert(flag==0);
                            V_guess = V_i(:,j);
                        end
                    end
                end

                S.Xi(:,i,k_ind) = S.Xi(:,i,k_ind) - S.wkpthf(q_ind)*(psi_q_set.*V_i)*S.occ_outer(:,q_ind_rd);
            end
        end
        M = S.psi_outer(:,:,k_ind)'*S.Xi(:,:,k_ind)*S.dx*S.dy*S.dz;
        % to ensure M is hermitian
        M = 0.5*(M+M');
        L = chol(-M);
        S.Xi(:,:,k_ind) = S.Xi(:,:,k_ind) * inv(L); % Do it efficiently
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

