function [x] = FD_FFT(S,rho)
% only work for periodic B.C.
if S.BC ~= 2
    error("only work for PBC");
end
rhs = -4*pi*(rho + S.b);
N1 = S.Nx;
N2 = S.Ny;
N3 = S.Nz;

f = reshape(rhs,N1,N2,N3);
f_hat = fftn(f);
u_hat = f_hat ./ S.d_hat;
u_hat(1) = 0; % this is because f_hat(1) = int {f_hat} = 0
x = ifftn(u_hat);
x = x(:);
x = x - dot(S.W,x)/sum(S.W);
end