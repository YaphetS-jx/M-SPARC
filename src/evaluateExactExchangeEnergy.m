function [S] = evaluateExactExchangeEnergy(S)
S.Eex = 0;
V_guess = rand(S.N,1);
for i = 1:S.Nev
    for j = 1:S.Nev
        rhs = S.psi_outer(:,i).*S.psi_outer(:,j);
        % For periodic case
%         gij = poissonSolve_FFT(S,rhs);

        % for dirichlet case
        f = poisson_RHS(rhs,S);
        [gij, flag] = pcg(-S.Lap_std,-f,1e-8,1000,S.LapPreconL,S.LapPreconU,V_guess);
        assert(flag==0);
        V_guess = gij;
        S.Eex = S.Eex + S.occ_outer(i)*S.occ_outer(j)*sum(rhs.*gij.*S.W);
    end
end

% S.Eex = (-1)*S.Eex;
S.Etotal = S.Etotal + 0.25*S.Eex;
fprintf(' Eex = %.8f\n', S.Eex);
fprintf(' Etot = %.8f\n', S.Etotal);
fprintf(2,' ------------------\n');

end