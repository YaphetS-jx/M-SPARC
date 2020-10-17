function Vexx = evaluateExactExchangePotential(S,X)
% t1 = tic;
Vexx = zeros(S.N,size(X,2));
% V_guess = rand(S.N,1);
for i = 1:size(X,2)
    for j = 1:S.Nev
        rhs = conj(S.psi_outer(:,j)).*X(:,i);
%         f = Poisson_RHS(rhs,S);
%         f = f - sum(f)/length(f);
%         V_ji = Pois_FFT_Periodic(f,S.w2,S.FDn,S.Nx,S.Ny,S.Nz,S.dx,S.dy,S.dz);
        V_ji = poissonSolve_FFT(S,rhs);
%         V_guess = V_ji;
        Vexx(:,i) = Vexx(:,i) - S.occ(j)*(V_ji.*S.psi_outer(:,j));
    end
end
% fprintf('Time taken by exact exchange matrix vector calculations is %.2f\n',toc(t1));

end
