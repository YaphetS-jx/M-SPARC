function [Etot,Et,Exc,Eelec] = ofdftEnergy(S)
rho = S.rho;
u = sqrt(rho);
% S.ofdft_lambda = 0.2;
S.ofdft_Cf = 0.3*((3*pi*pi)^(2/3));
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

[DL11,DL22,DL33,DG1,DG2,DG3] = blochLaplacian_1d(S,[0,0,0]);
Hu = lapVec(DL11,DL22,DL33,DG1,DG2,DG3,u,S);
Et2 = -0.5 * dot(u,Hu) * S.dV;
Et = Et1 + S.ofdft_lambda * Et2;

% Total free energy
Etot =  Et + Exc + Eelec - S.Eself + S.E_corr;
end
