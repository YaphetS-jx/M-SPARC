function [Etot,Eband,Exc,Exc_dc,Eelec_dc,Eent,EigVal,occ] = ksEnergy(S)
%% Calculating Kohn-Sham energy using electron density from OFDFT
% Please change the variables before using

S.Nev = round(floor(S.Nelectron / 2) * 1.2 + 5);
S.nkpt = S.nsc_nkpt;
S.kptshift = S.nsc_kptshift;

[S] = Generate_kpts(S);
%% codes
S.Atom = calculate_nloc_projector(S);

% Electrostatic potential
S = poissonSolve(S, S.poisson_tol, 0);

% Exchange-correlation potential
S = exchangeCorrelationPotential(S);

% Effective potential
S.Veff = real(bsxfun(@plus,S.phi,S.Vxc));


rng('default'); % Initialize random number generator
rng(1); % Specify the seed to be 1
S.psi = rand(S.N,S.Nev,S.tnkpt*S.nspin)-0.5;
S.upper_bound_guess_vecs = zeros(S.N,S.tnkpt*S.nspin);
S.EigVal = zeros(S.Nev,S.tnkpt*S.nspin);


% Spectrum bounds and filter cutoff for Chebyshev filtering
bup = zeros(S.tnkpt*S.nspin,1);
a0 = zeros(S.tnkpt*S.nspin,1);
lambda_cutoff = zeros(S.tnkpt*S.nspin,1);

for count = 1:10
    [S.upper_bound_guess_vecs,S.psi,S.EigVal,a0,bup,lambda_cutoff] = ...
        eigSolver(S,count,S.upper_bound_guess_vecs,S.psi,S.EigVal,a0,bup,lambda_cutoff);
    fprintf("ChefSI %d done\n", count);
end

S = occupations(S);

[Etot,Eband,Exc,Exc_dc,Eelec_dc,Eent] = evaluateTotalEnergy(S);
EigVal = S.EigVal;
occ = S.occ;
end



function [S] = Generate_kpts(S)
	nkpt = S.nkpt;
	if (S.BCx == 1 && nkpt(1) > 1)
		error(' nkpt cannot be greater than 1 in Dirichlet boundary direction (x)');
	end
	if (S.BCy == 1 && nkpt(2) > 1)
		error(' nkpt cannot be greater than 1 in Dirichlet boundary direction (y)');
	end
	if (S.BCz == 1 && nkpt(3) > 1)
		error(' nkpt cannot be greater than 1 in Dirichlet boundary direction (z)');
	end

	% Monkhorst-pack grid for Brillouin zone sampling
	MPG_typ1 = @(nkpt) (2*(1:nkpt) - nkpt - 1)/2; % MP grid points for infinite group order
	MPG_typ2 = @(nkpt) (0:nkpt-1); % MP grid points for finite group order

	if S.cell_typ < 3
		kptgrid_x = (1/nkpt(1)) * MPG_typ1(nkpt(1));
		kptgrid_y = (1/nkpt(2)) * MPG_typ1(nkpt(2));
		kptgrid_z = (1/nkpt(3)) * MPG_typ1(nkpt(3));
		sumx = 0;
		sumy = 0; 
		sumz = 0;
		% shift kpoint grid 
		kptgrid_x = kptgrid_x + S.kptshift(1) * (1/nkpt(1));
		kptgrid_y = kptgrid_y + S.kptshift(2) * (1/nkpt(2));
		kptgrid_z = kptgrid_z + S.kptshift(3) * (1/nkpt(3));
	
		% map k-points back to BZ
		temp_epsilon = eps; % include the right boundary k-points instead of left
		kptgrid_x = mod(kptgrid_x + 0.5 - temp_epsilon, 1) - 0.5 + temp_epsilon;
		kptgrid_y = mod(kptgrid_y + 0.5 - temp_epsilon, 1) - 0.5 + temp_epsilon;
		kptgrid_z = mod(kptgrid_z + 0.5 - temp_epsilon, 1) - 0.5 + temp_epsilon;
	elseif (S.cell_typ == 3 || S.cell_typ == 4 || S.cell_typ == 5)
		kptgrid_x = (1/nkpt(1)) * MPG_typ1(nkpt(1));
		kptgrid_y = (1/nkpt(2)) * MPG_typ2(nkpt(2));
		kptgrid_z = (1/nkpt(3)) * MPG_typ1(nkpt(3));
		sumx = 0;
		sumy = nkpt(2); 
		sumz = 0;
	end    
	
	% Scale kpoints
	kptgrid_x = (2*pi/S.L1) * kptgrid_x;
	kptgrid_y = (2*pi/S.L2) * kptgrid_y;
	kptgrid_z = (2*pi/S.L3) * kptgrid_z;

	[kptgrid_X, kptgrid_Y, kptgrid_Z] = ndgrid(kptgrid_x,kptgrid_y,kptgrid_z);
	kptgrid = [reshape(kptgrid_X,[],1),reshape(kptgrid_Y,[],1),reshape(kptgrid_Z,[],1)];
% 	disp(' kpoint grid before symmetry:');
% 	disp(kptgrid);
	
	tnkpt = prod(nkpt);
	wkpt = ones(tnkpt,1)/tnkpt;% weights for k-points
	TOL = 1e-8;
	% Time-Reversal Symmetry to reduce k-points
	if S.TimeRevSym == 1
		Ikpt = zeros(tnkpt,1);
		Ikpt_rev = zeros(tnkpt,1);
		for ii = 1:tnkpt
			for jj = ii+1:tnkpt
				if (abs(kptgrid(ii,1) + kptgrid(jj,1) - sumx) < TOL) && (abs(kptgrid(ii,2) + kptgrid(jj,2) - sumy) < TOL) && (abs(kptgrid(ii,3) + kptgrid(jj,3) - sumz) < TOL)
					Ikpt(ii) = 1;
					Ikpt_rev(jj) = 1;
				end
			end
		end
		Ikpt = Ikpt>0.5;
		Ikpt_rev = Ikpt_rev>0.5;
		wkpt(Ikpt_rev) = 2*wkpt(Ikpt_rev);
		kptgrid = kptgrid(~Ikpt,:);
		wkpt = wkpt(~Ikpt);
		tnkpt = size(wkpt,1);
	end

% 	disp(' kpoint grid after symmetry:');	
% 	disp(kptgrid);
	% Store into the structure
	S.kptgrid = kptgrid;
	S.tnkpt   = tnkpt;
	S.wkpt    = wkpt;
end