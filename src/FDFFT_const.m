function S = FDFFT_const(S)
% only for periodic B.C.

w2 = S.w2;
FDn = S.FDn;
N1 = S.Nx;
N2 = S.Ny;
N3 = S.Nz;
dx = S.dx;
dy = S.dy;
dz = S.dz;

dx2 = dx*dx; dy2 = dy*dy; dz2 = dz*dz;

w2_x = w2 / dx2;
w2_y = w2 / dy2;
w2_z = w2 / dz2;

% FD approximationi of d_hat = G^2
% alpha follows conjugate even space
count = 1;
d_hat = zeros(N1,N2,N3);
% G2 = zeros(N1,N2,N3);
w2_diag = w2_x(1) + w2_y(1) +w2_z(1);
for k3 = [1:floor(N3/2)+1, floor(-N3/2)+2:0]
    for k2 = [1:floor(N2/2)+1, floor(-N2/2)+2:0]
        for k1 = [1:floor(N1/2)+1, floor(-N1/2)+2:0]
% 			G2(count) = ((k1-1)*2*pi/L1)^2 + ((k2-1)*2*pi/L2)^2 + ((k3-1)*2*pi/L3)^2;
			d_hat(count) = -w2_diag;
            for p = 1:FDn
                d_hat(count) = d_hat(count) - 2 * ...
                    (  cos(2*pi*(k1-1)*p/N1)*w2_x(p+1) ...
                     + cos(2*pi*(k2-1)*p/N2)*w2_y(p+1) ...
                     + cos(2*pi*(k3-1)*p/N3)*w2_z(p+1));
            end
            count = count + 1;
        end
    end
end
d_hat(1) = 1;
S.d_hat = -1*d_hat;
end
