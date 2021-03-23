function [S] = NLCG_Teter(S,u)
[DL11,DL22,DL33,DG1,DG2,DG3] = blochLaplacian_1d(S,S.kptgrid(1,:));
S.ofdft_phi_flag = 0;
% options = optimset('fminsearch');
options = optimset('fminbnd');
options.TolX = 1E-2;
% options.TolFun = 1E-6;

outfname = S.outfname;
fileID = fopen(outfname,'a');
fprintf(fileID,'=====================================================================\n');
fprintf(fileID,'             Nonlinear Conjugate Gradient (NLCG#%d)                  \n',S.Relax_iter);
fprintf(fileID,'=====================================================================\n');
fprintf(fileID,'Iteration     s       iter         Error                 Timing (sec)\n');
fclose(fileID);

tic_nlcg = tic;
nlcg_runtime = 0;

i = 0; 
k = 0;
imax = 1500; 
tol1 = 1E-14 * S.N;
% tol1 = 1E-7;
% deltaNew = tol1 + 1;

[F,S] = Hx(S,DL11,DL22,DL33,DG1,DG2,DG3,u);
eta = dot(F,u) * S.dV / S.Nelectron;
r = -2 * (F - eta * u);
% not sure here
% d = r;
d = 0 * r;

% find s
OFDFTEnergyEvaluator = @(s) ofdft_find_mins(S,u,s,r);
% [s,~,exitflag] = fminsearch(OFDFTEnergyEvaluator,0.2,options);
[s,~,exitflag,output] = fminbnd(OFDFTEnergyEvaluator,0,1,options);
if exitflag ~= 1
%     error("fminsearch not converged\n");
    error("fminbnd not converged\n");
end

u = u + s * r;
u = sqrt(S.Nelectron / (dot(u,u) * S.dV)) * u;
u = abs(u);
% dold = r;
rold = r;

while i < imax
    [F,S] = Hx(S,DL11,DL22,DL33,DG1,DG2,DG3,u);
    eta = dot(F,u) * S.dV / S.Nelectron;
    r = -2 * (F - eta * u);
    deltaNew = dot(r,r);
    fprintf("iter %d s %.6f iter %d error %.3e\n", i+1,s,output.iterations,deltaNew);
    
    if deltaNew < tol1
        break;
    end
    
    nlcg_runtime = nlcg_runtime + toc(tic_nlcg);
    fileID = fopen(outfname,'a');
    fprintf(fileID,'%-6d       %.6f  %2d   %.3E                  %.3f\n', ...
					i, s, output.iterations, deltaNew, nlcg_runtime);
    fclose(fileID);
    tic_nlcg = tic;
    
    
    v1 = dot(rold,r);
    v2 = dot(rold,rold);
    xi = (deltaNew - v1) / v2;
    if k == 30 || xi <= 0
        d = r;
        k = 0;
    else
        d = xi * d + r;
    end
    % find s
    OFDFTEnergyEvaluator = @(s) ofdft_find_mins(S,u,s,d);
%     [s,~,exitflag] = fminsearch(OFDFTEnergyEvaluator, 0.2,options);
    [s,~,exitflag,output] = fminbnd(OFDFTEnergyEvaluator,0,1,options);
    
    if exitflag ~= 1
%         error("fminsearch not converged\n");
        error("fminbnd not converged\n");
    end
    
    u = u + s * r;
    u = sqrt(S.Nelectron / (dot(u,u) * S.dV)) * u;
    u = abs(u);
%     dold = d;
    rold = r;
    k = k + 1;
    i = i + 1;
end

fprintf('\n Finished NLCG in %d steps!\n', i+1);
% fprintf('\n dot(u,u)*dV = %.4f\n', dot(u,u)*S.dV);

[S.Etotal,S.Et,S.Exc] = ofdftTotalEnergy(S,u);

nlcg_runtime = nlcg_runtime + toc(tic_nlcg);
fileID = fopen(outfname,'a');
fprintf(fileID,'%-6d       %.6f  %2d   %.3E                  %.3f\n', ...
                i, s,  output.iterations, deltaNew, nlcg_runtime);
fclose(fileID);
end




function [F,S] = Hx(S,DL11,DL22,DL33,DG1,DG2,DG3,u)

F = -0.5*S.ofdft_lambda*(lapVec(DL11,DL22,DL33,DG1,DG2,DG3,u,S));
rho = u.^2;
S.rho = rho;
% phi
% if S.ofdft_phi_flag == 0
%     S = poissonSolve(S, S.poisson_tol, 0);
%     S.ofdft_phi_flag = 1;
% else
%     S = poissonSolve(S, S.poisson_tol, 1);
% end

S.phi = FD_FFT(S,rho);

% Vxc
S = exchangeCorrelationPotential(S);
Vk = (5/3)*S.ofdft_Cf*(rho.^(2/3));
Veff = S.phi + S.Vxc + Vk;

F = F + Veff.*u;

end


function [Eatm] = ofdft_find_mins(S,u,s,d)
u = u + s .* d;
u = sqrt(S.Nelectron / (dot(u,u) * S.dV)) * u;
u = abs(u);
S.rho = u.^2;

% S = poissonSolve(S, S.poisson_tol, 1);

S.phi = FD_FFT(S,S.rho);

Etot = ofdftTotalEnergy(S,u);
Eatm = Etot/S.n_atm;
end

