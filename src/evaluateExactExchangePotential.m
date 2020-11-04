function Vexx = evaluateExactExchangePotential(S,X)
% t1 = tic;
Vexx = zeros(S.N,size(X,2));
V_guess = rand(S.N,1);
for i = 1:size(X,2)
    for j = 1:S.Nev
        rhs = conj(S.psi_outer(:,j)).*X(:,i);
        f = Poisson_RHS(rhs,S);
        [V_ji, flag]= gmres(S.Lap_std,f,50,1e-8,1000,S.LapPreconL,S.LapPreconU,V_guess);
        assert(flag == 0);
        V_guess = V_ji;
        Vexx(:,i) = Vexx(:,i) - S.occ(j)*(V_ji.*S.psi_outer(:,j));
    end
end
% fprintf('Time taken by exact exchange matrix vector calculations is %.2f\n',toc(t1));
end