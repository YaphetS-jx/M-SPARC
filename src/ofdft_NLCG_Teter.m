function [S] = ofdft_NLCG_Teter(S,u)
% @brief   Orbital-Free DFT nonlinear conjugate Teter solver
%
% @authors  Xin Jing <xjing30@gatech.edu>
%           Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
% 
% @param u           sqrt of electron density rho
%
% @copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech
%========================================================================================

[DL11,DL22,DL33,DG1,DG2,DG3] = blochLaplacian_1d(S,S.kptgrid(1,:));
options = optimset('fminbnd');
options.TolX = 1E-2;

outfname = S.outfname;
fileID = fopen(outfname,'a');
fprintf(fileID,'=====================================================================\n');
fprintf(fileID,'                Orbital-Free DFT NLCG (OFDFT-NLCG#%d)                \n',S.Relax_iter);
fprintf(fileID,'=====================================================================\n');
fprintf(fileID,'Iteration      Free Energy (Ha/atom)   NLCG Error        Timing (sec)\n');
fclose(fileID);

tic_nlcg = tic;

i = 0; 
k = 0;
imax = 1500; 
tol1 = S.ofdft_tol^2 * S.N;

[F,S] = Hx(S,DL11,DL22,DL33,DG1,DG2,DG3,u);
eta = dot(F,u) * S.dV / S.Nelectron;
r = -2 * (F - eta * u);
d = zeros(S.N,1);

% find s
OFDFTEnergyEvaluator = @(s) ofdft_find_mins(S,u,s,r);
[s,fval,exitflag,output] = fminbnd(OFDFTEnergyEvaluator,0,1,options);
if exitflag ~= 1
    error('fminbnd not converged within %d iterations\n',output.iterations);
end

u = u + s * r;
u = sqrt(S.Nelectron / (dot(u,u) * S.dV)) * u;
u = abs(u);
rold = r;

while i < imax
    [F,S] = Hx(S,DL11,DL22,DL33,DG1,DG2,DG3,u);
    eta = dot(F,u) * S.dV / S.Nelectron;
    r = -2 * (F - eta * u);
    deltaNew = dot(r,r);
    fprintf(" iter%-5d  s %.6f sit %d Energy(Ha/atom) %.6E   error %.3E\n", i+1,s,output.iterations, fval,sqrt(deltaNew/S.N));
    
    fileID = fopen(outfname,'a');
    fprintf(fileID,'%-6d        %18.10E       %.3E         %.3f\n', ...
					i+1, fval, sqrt(deltaNew/S.N), toc(tic_nlcg));
    fclose(fileID);
    tic_nlcg = tic;
    
    if deltaNew < tol1
        break;
    end
    
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
    [s,fval,exitflag,output] = fminbnd(OFDFTEnergyEvaluator,0,1,options);
    if exitflag ~= 1
        error('fminbnd not converged within %d iterations\n',output.iterations);
    end
    
    u = u + s * r;
%     u = u + s * d;
    u = sqrt(S.Nelectron / (dot(u,u) * S.dV)) * u;
    u = abs(u);
    rold = r;
    k = k + 1;
    i = i + 1;
end

fprintf('\n Finished NLCG in %d steps!\n', i+1);

[S.Etotal,S.Et,S.Exc] = ofdftTotalEnergy(S,u);
end

% function to compute (-lambda/5*lap + phi + Vxc + Vk) (x)
function [F,S] = Hx(S,DL11,DL22,DL33,DG1,DG2,DG3,u)
% -lambda/5*lap(x)
F = -0.5*S.ofdft_lambda*(lapVec(DL11,DL22,DL33,DG1,DG2,DG3,u,S));
rho = u.^2;
S.rho = rho;
% phi
S = poissonSolve(S, S.poisson_tol, 0);
% Vxc
S = exchangeCorrelationPotential(S);
% Vk
Vk = (5/3)*S.ofdft_Cf*(rho.^(2/3));
Veff = S.phi + S.Vxc + Vk;
F = F + Veff.*u;

end


function [Eatm] = ofdft_find_mins(S,u,s,d)
u = u + s .* d;
u = sqrt(S.Nelectron / (dot(u,u) * S.dV)) * u;
u = abs(u);
S.rho = u.^2;
% phi
S = poissonSolve(S, S.poisson_tol, 0);
Etot = ofdftTotalEnergy(S,u);
Eatm = Etot/S.n_atm;
end


function [Etot,Et,Exc] = ofdftTotalEnergy(S,u)
rho = S.rho;
% Exchange-correlation energy
% Check if density is too small
INDX_zerorho = (rho < S.xc_rhotol);
rho(INDX_zerorho) = S.xc_rhotol;

if S.nspin == 1
	if S.xc == 0 % LDA_PW
		C2 = 0.73855876638202 ; % constant for exchange energy
		%rho = rho+(1e-50) ; % to avoid divide by zero error
		p = 1 ;
		A = 0.031091 ;
		alpha1 = 0.21370 ;
		beta1 = 7.5957 ;
		beta2 = 3.5876 ;
		beta3 = 1.6382 ;
		beta4 = 0.49294 ;
		CEnergyPotential = (0.75./(pi*rho)).^(1/3) ;
		CEnergyPotential = -2*A*(1+alpha1*CEnergyPotential).*log(1+1./(2*A*( beta1*(CEnergyPotential.^0.5) ...
		   + beta2*CEnergyPotential + beta3*(CEnergyPotential.^1.5) + beta4*(CEnergyPotential.^(p+1.0))))) ;
		%rho = rho-(1e-50) ;
		Exc = sum(CEnergyPotential.*rho.*S.W) - C2*sum((rho.^(4/3)).*S.W) ;
	elseif S.xc == 1 % LDA_PZ
		A = 0.0311;
		B = -0.048 ;
		C = 0.002 ;
		D = -0.0116 ;
		gamma1 = -0.1423 ;
		beta1 = 1.0529 ;
		beta2 = 0.3334 ;
		C2 = 0.73855876638202;
		%rho = rho+(1e-50) ; % to avoid divide by zero error
		CEnergyPotential = (0.75./(pi*rho)).^(1/3) ;
		islt1 = (CEnergyPotential < 1.0);
		CEnergyPotential(islt1) = A * log(CEnergyPotential(islt1)) + B ...
		   + C * CEnergyPotential(islt1) .* log(CEnergyPotential(islt1)) ...
		   + D * CEnergyPotential(islt1);
		CEnergyPotential(~islt1) = gamma1./(1.0+beta1*sqrt(CEnergyPotential(~islt1))+beta2*CEnergyPotential(~islt1));
		%rho = rho-(1e-50) ;
		Exc = sum(CEnergyPotential.*rho.*S.W) - C2*sum((rho.^(4/3)).*S.W) ;
	elseif S.xc == 2
		Exc = sum(S.e_xc.*rho.*S.W);
	end
	% Exchange-correlation energy double counting correction
% 	Exc_dc = sum(S.Vxc.*rho.*S.W) ;
end

% option A as in PETSC code
Eelec = 0.5*sum((S.b+S.rho(:,1)).*S.phi.*S.W);

Et1 = S.ofdft_Cf*sum(rho.^(5/3))*S.dV;

[DL11,DL22,DL33,DG1,DG2,DG3] = blochLaplacian_1d(S,S.kptgrid(1,:));
Hu = lapVec(DL11,DL22,DL33,DG1,DG2,DG3,u,S);
Et2 = -0.5 * dot(u,Hu) * S.dV;
Et = Et1 + S.ofdft_lambda * Et2;

% Total free energy
Etot =  Et + Exc + Eelec - S.Eself + S.E_corr;
end


