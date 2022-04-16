function stress = evaluateStress(S)
% @ brief    Function to calculate stress in periodic systems (O(N^3))
% @ authors
%          Abhiraj Sharma <asharma424@gatech.edu>
%          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
%
% @ references
%             "On the calculation of the stress tensor in real-space Kohn-Sham
%              density functional theory (Sharma et. al. 2018)"
%
% @copyright (c) 2019 Material Physics & Mechanics Group, Georgia Tech
%===============================================================================
stress = zeros(3,3);
S_T = transpose(S.lat_uvec);

% Stress contribution from Kinetic component
ks = 1;
for spin = 1:S.nspin
	for kpt = 1:S.tnkpt
		kpt_vec = S.kptgrid(kpt,:);

		Dpsi_x = blochGradient(S,kpt_vec,1)*S.psi(:,:,ks);
		Dpsi_y = blochGradient(S,kpt_vec,2)*S.psi(:,:,ks);
		Dpsi_z = blochGradient(S,kpt_vec,3)*S.psi(:,:,ks);

		TDpsi_1 = S.grad_T(1,1)*Dpsi_x + S.grad_T(2,1)*Dpsi_y + S.grad_T(3,1)*Dpsi_z;
		TDpsi_2 = S.grad_T(1,2)*Dpsi_x + S.grad_T(2,2)*Dpsi_y + S.grad_T(3,2)*Dpsi_z;
		TDpsi_3 = S.grad_T(1,3)*Dpsi_x + S.grad_T(2,3)*Dpsi_y + S.grad_T(3,3)*Dpsi_z;
		TDcpsi_1 = conj(TDpsi_1);
		TDcpsi_2 = conj(TDpsi_2);
		TDcpsi_3 = conj(TDpsi_3);

		stress(1,1) = stress(1,1) + real(-S.occfac * S.wkpt(kpt) * (transpose(S.W) * ... 
					  (TDcpsi_1.*TDpsi_1)) * S.occ(:,ks));
		stress(1,2) = stress(1,2) + real(-S.occfac * S.wkpt(kpt) * (transpose(S.W) * ... 
					  (TDcpsi_1.*TDpsi_2)) * S.occ(:,ks));
		stress(1,3) = stress(1,3) + real(-S.occfac * S.wkpt(kpt) * (transpose(S.W) * ... 
					  (TDcpsi_1.*TDpsi_3)) * S.occ(:,ks));    
		stress(2,1) = stress(2,1) + real(-S.occfac * S.wkpt(kpt) * (transpose(S.W) * ... 
					  (TDcpsi_2.*TDpsi_1)) * S.occ(:,ks));
		stress(2,2) = stress(2,2) + real(-S.occfac * S.wkpt(kpt) * (transpose(S.W) * ... 
					  (TDcpsi_2.*TDpsi_2)) * S.occ(:,ks));
		stress(2,3) = stress(2,3) + real(-S.occfac * S.wkpt(kpt) * (transpose(S.W) * ... 
					  (TDcpsi_2.*TDpsi_3)) * S.occ(:,ks));
		stress(3,1) = stress(3,1) + real(-S.occfac * S.wkpt(kpt) * (transpose(S.W) * ... 
					  (TDcpsi_3.*TDpsi_1)) * S.occ(:,ks));
		stress(3,2) = stress(3,2) + real(-S.occfac * S.wkpt(kpt) * (transpose(S.W) * ... 
					  (TDcpsi_3.*TDpsi_2)) * S.occ(:,ks));
		stress(3,3) = stress(3,3) + real(-S.occfac * S.wkpt(kpt) * (transpose(S.W) * ... 
					  (TDcpsi_3.*TDpsi_3)) * S.occ(:,ks));
		
		ks = ks + 1;           
	end
end

fprintf('\n[\b"Kinetic Stress in GPa"\n\n\n]\b');
disp(stress/(S.Jacb*S.L1*S.L2*S.L3)*2.94210119*(10^4));
stress_temp = stress;
    
% Stress contribution from exchange-correlation and energy terms from electrostatics
Drho_x = S.grad_1 * S.rho;
Drho_y = S.grad_2 * S.rho;
Drho_z = S.grad_3 * S.rho;

if S.nspin == 1
	for alpha = 1:3
		for beta = 1:3
			Drho_alpha = S.grad_T(1,alpha)*Drho_x + S.grad_T(2,alpha)*Drho_y + S.grad_T(3,alpha)*Drho_z ;
			Drho_beta = S.grad_T(1,beta)*Drho_x + S.grad_T(2,beta)*Drho_y + S.grad_T(3,beta)*Drho_z ;
			stress(alpha,beta) = stress(alpha,beta) + (alpha == beta) * ( S.Exc - sum(S.W .* S.Vxc .* S.rho) )  + ...
								 -sum(S.W' * (S.dvxcdgrho .* Drho_alpha .* Drho_beta)) +...
								 (alpha == beta) * ( 0.5 * sum( S.W .* (S.b - S.rho) .* S.phi ) - S.Eself + S.E_corr )  ;
		end
	end
	
else
	for alpha = 1:3
		for beta = 1:3
			Drho_alpha = S.grad_T(1,alpha)*Drho_x + S.grad_T(2,alpha)*Drho_y + S.grad_T(3,alpha)*Drho_z ;
			Drho_beta = S.grad_T(1,beta)*Drho_x + S.grad_T(2,beta)*Drho_y + S.grad_T(3,beta)*Drho_z ;
			stress(alpha,beta) = stress(alpha,beta) + (alpha == beta) * ( S.Exc - sum(S.W'*(S.Vxc.*S.rho(:,2:3))))  + ...
								-sum(S.W' * (S.dvxcdgrho .* Drho_alpha .* Drho_beta)) +...
								 (alpha == beta) * ( 0.5 * sum( S.W .* (S.b - S.rho(:,1)) .* S.phi ) - S.Eself + S.E_corr )  ;
		end
	end
end

fprintf('\n[\b"XC Stress in GPa"\n\n\n]\b');
disp((stress-stress_temp)/(S.Jacb*S.L1*S.L2*S.L3)*2.94210119*(10^4));
stress_temp = stress;

% Stress contribution from remaining terms in electrostatics
Dphi_x = S.grad_1*(S.phi);
Dphi_y = S.grad_2*(S.phi);
Dphi_z = S.grad_3*(S.phi);

TDphi_1 = S.grad_T(1,1)*Dphi_x + S.grad_T(2,1)*Dphi_y + S.grad_T(3,1)*Dphi_z ;
TDphi_2 = S.grad_T(1,2)*Dphi_x + S.grad_T(2,2)*Dphi_y + S.grad_T(3,2)*Dphi_z ;
TDphi_3 = S.grad_T(1,3)*Dphi_x + S.grad_T(2,3)*Dphi_y + S.grad_T(3,3)*Dphi_z ;
stress(1,1) = stress(1,1) + (1/4/pi) * sum( TDphi_1.*TDphi_1.*S.W);
stress(1,2) = stress(1,2) + (1/4/pi) * sum( TDphi_1.*TDphi_2.*S.W);
stress(1,3) = stress(1,3) + (1/4/pi) * sum( TDphi_1.*TDphi_3.*S.W);
stress(2,1) = stress(2,1) + (1/4/pi) * sum( TDphi_2.*TDphi_1.*S.W);
stress(2,2) = stress(2,2) + (1/4/pi) * sum( TDphi_2.*TDphi_2.*S.W);
stress(2,3) = stress(2,3) + (1/4/pi) * sum( TDphi_2.*TDphi_3.*S.W);
stress(3,1) = stress(3,1) + (1/4/pi) * sum( TDphi_3.*TDphi_1.*S.W);
stress(3,2) = stress(3,2) + (1/4/pi) * sum( TDphi_3.*TDphi_2.*S.W);
stress(3,3) = stress(3,3) + (1/4/pi) * sum( TDphi_3.*TDphi_3.*S.W);

%psdfilepath = sprintf('%s/PseudopotFiles',S.inputfile_path);
count_typ = 1;
count_typ_atms = 1;
for JJ_a = 1:S.n_atm % loop over all the atoms
	% Atom position
	x0 = S.Atoms(JJ_a,1);
	y0 = S.Atoms(JJ_a,2);
	z0 = S.Atoms(JJ_a,3);
	% Note the S.dx, S.dy, S.dz terms are to ensure the image rb-region overlap w/ fund. domain
	if S.BCx == 0
		n_image_xl = floor((S.Atoms(JJ_a,1) + S.Atm(count_typ).rb_x)/S.L1);
		n_image_xr = floor((S.L1 - S.Atoms(JJ_a,1)+S.Atm(count_typ).rb_x-S.dx)/S.L1);
	else
		n_image_xl = 0;
		n_image_xr = 0;
	end
	
	if S.BCy == 0
		n_image_yl = floor((S.Atoms(JJ_a,2) + S.Atm(count_typ).rb_y)/S.L2);
		n_image_yr = floor((S.L2 - S.Atoms(JJ_a,2)+S.Atm(count_typ).rb_y-S.dy)/S.L2);
	else
		n_image_yl = 0;
		n_image_yr = 0;
	end
	
	if S.BCz == 0
		n_image_zl = floor((S.Atoms(JJ_a,3) + S.Atm(count_typ).rb_z)/S.L3);
		n_image_zr = floor((S.L3 - S.Atoms(JJ_a,3)+S.Atm(count_typ).rb_z-S.dz)/S.L3);
	else
		n_image_zl = 0;
		n_image_zr = 0;
	end
	
	% Total No. of images of atom JJ_a (including atom JJ_a)
	n_image_total = (n_image_xl+n_image_xr+1) * (n_image_yl+n_image_yr+1) * (n_image_zl+n_image_zr+1);
	% Find the coordinates for all the images
	xx_img = [-n_image_xl : n_image_xr] * S.L1 + x0;
	yy_img = [-n_image_yl : n_image_yr] * S.L2 + y0;
	zz_img = [-n_image_zl : n_image_zr] * S.L3 + z0;
	[XX_IMG_3D,YY_IMG_3D,ZZ_IMG_3D] = ndgrid(xx_img,yy_img,zz_img);

	% Loop over all image(s) of atom JJ_a (including atom JJ_a)
	for count_image = 1:n_image_total

		% Atom position of the image
		x0_i = XX_IMG_3D(count_image);
		y0_i = YY_IMG_3D(count_image);
		z0_i = ZZ_IMG_3D(count_image);

		% Indices of closest grid point to atom
		pos_ii = round((x0_i-S.xin) / S.dx) + 1;
		pos_jj = round((y0_i-S.yin) / S.dy) + 1;
		pos_kk = round((z0_i-S.zin) / S.dz) + 1;

		% Starting and ending indices of b-region
		ii_s = pos_ii - ceil(S.Atm(count_typ).rb_x/S.dx+0.5);
		ii_e = pos_ii + ceil(S.Atm(count_typ).rb_x/S.dx+0.5);
		jj_s = pos_jj - ceil(S.Atm(count_typ).rb_y/S.dy+0.5);
		jj_e = pos_jj + ceil(S.Atm(count_typ).rb_y/S.dy+0.5);
		kk_s = pos_kk - ceil(S.Atm(count_typ).rb_z/S.dz+0.5);
		kk_e = pos_kk + ceil(S.Atm(count_typ).rb_z/S.dz+0.5);

		% Check if the b-region is inside the domain in Dirichlet BC
		% direction
		%isInside = (S.BCx == 0 || (S.BCx == 1 && (ii_s>1) && (ii_e<S.Nx))) && ...
		%   (S.BCy == 0 || (S.BCy == 1 && (jj_s>1) && (jj_e<S.Ny))) && ...
		%   (S.BCz == 0 || (S.BCz == 1 && (kk_s>1) && (kk_e<S.Nz)));
		% assert(isInside,'Error: Atom too close to boundary for b calculation');
		ii_s = max(ii_s,1);
		ii_e = min(ii_e,S.Nx);
		jj_s = max(jj_s,1);
		jj_e = min(jj_e,S.Ny);
		kk_s = max(kk_s,1);
		kk_e = min(kk_e,S.Nz);

		xx = S.xin + (ii_s-2*S.FDn-1:ii_e+2*S.FDn-1)*S.dx;% - x0_i;
		yy = S.yin + (jj_s-2*S.FDn-1:jj_e+2*S.FDn-1)*S.dy;% - y0_i;
		zz = S.zin + (kk_s-2*S.FDn-1:kk_e+2*S.FDn-1)*S.dz;% - z0_i;
		[XX_3D,YY_3D,ZZ_3D] = ndgrid(xx,yy,zz);

		% Find distances
		dd = calculateDistance(XX_3D,YY_3D,ZZ_3D,x0_i,y0_i,z0_i,S);

		% Pseudopotential at grid points through interpolation
		V_PS = zeros(size(dd));
		IsLargeThanRmax = dd > S.Atm(count_typ).r_grid_vloc(end);
		V_PS(IsLargeThanRmax) = -S.Atm(count_typ).Z;
		V_PS(~IsLargeThanRmax) = interp1(S.Atm(count_typ).r_grid_vloc, S.Atm(count_typ).r_grid_vloc.*S.Atm(count_typ).Vloc, dd(~IsLargeThanRmax), 'spline');

		V_PS = V_PS./dd;
		V_PS(dd<S.Atm(count_typ).r_grid_vloc(2)) = S.Atm(count_typ).Vloc(1); % WARNING

		% Reference potential at grid points
		rc_ref = S.rc_ref; % WARNING: Might need smaller if pseudocharges overlap
		V_PS_ref = zeros(size(dd));
		I_ref = dd<rc_ref;
		V_PS_ref(~I_ref) = -(S.Atm(count_typ).Z)./dd(~I_ref);
		V_PS_ref(I_ref) = -S.Atm(count_typ).Z*(9*dd(I_ref).^7-30*rc_ref*dd(I_ref).^6 ...
			+28*rc_ref*rc_ref*dd(I_ref).^5-14*(rc_ref^5)*dd(I_ref).^2+12*rc_ref^7)/(5*rc_ref^8);

		% Pseudocharge density
		II = 1+S.FDn : size(V_PS,1)-S.FDn;
		JJ = 1+S.FDn : size(V_PS,2)-S.FDn;
		KK = 1+S.FDn : size(V_PS,3)-S.FDn;
		
		% Calculate bJ and bJ_ref
		bJ = pseudochargeDensity_atom(V_PS,II,JJ,KK,xx(1),S);
		bJ_ref = pseudochargeDensity_atom(V_PS_ref,II,JJ,KK,xx(1),S);

		bJ = (-1/(4*pi))*bJ;
		bJ_ref = (-1/(4*pi))*bJ_ref;

		% Calculate the gradient of pseudocharges
		dbJ_x = zeros(size(V_PS)); dbJ_y = zeros(size(V_PS)); dbJ_z = zeros(size(V_PS));
		dbJ_ref_x = zeros(size(V_PS)); dbJ_ref_y = zeros(size(V_PS)); dbJ_ref_z = zeros(size(V_PS));
		dVJ_x = zeros(size(V_PS)); dVJ_y = zeros(size(V_PS)); dVJ_z = zeros(size(V_PS));
		dVJ_ref_x = zeros(size(V_PS)); dVJ_ref_y = zeros(size(V_PS)); dVJ_ref_z = zeros(size(V_PS));
		II = 1+2*S.FDn : size(V_PS,1)-2*S.FDn;
		JJ = 1+2*S.FDn : size(V_PS,2)-2*S.FDn;
		KK = 1+2*S.FDn : size(V_PS,3)-2*S.FDn;
		for p = 1:S.FDn
			dbJ_x(II,JJ,KK) = dbJ_x(II,JJ,KK) + S.w1(p+1)/S.dx*(bJ(II+p,JJ,KK)-bJ(II-p,JJ,KK));
			dbJ_y(II,JJ,KK) = dbJ_y(II,JJ,KK) + S.w1(p+1)/S.dy*(bJ(II,JJ+p,KK)-bJ(II,JJ-p,KK));
			dbJ_z(II,JJ,KK) = dbJ_z(II,JJ,KK) + S.w1(p+1)/S.dz*(bJ(II,JJ,KK+p)-bJ(II,JJ,KK-p));
			dbJ_ref_x(II,JJ,KK) = dbJ_ref_x(II,JJ,KK) + S.w1(p+1)/S.dx*(bJ_ref(II+p,JJ,KK)-bJ_ref(II-p,JJ,KK));
			dbJ_ref_y(II,JJ,KK) = dbJ_ref_y(II,JJ,KK) + S.w1(p+1)/S.dy*(bJ_ref(II,JJ+p,KK)-bJ_ref(II,JJ-p,KK));
			dbJ_ref_z(II,JJ,KK) = dbJ_ref_z(II,JJ,KK) + S.w1(p+1)/S.dz*(bJ_ref(II,JJ,KK+p)-bJ_ref(II,JJ,KK-p));
			dVJ_x(II,JJ,KK) = dVJ_x(II,JJ,KK) + S.w1(p+1)/S.dx*(V_PS(II+p,JJ,KK)-V_PS(II-p,JJ,KK));
			dVJ_y(II,JJ,KK) = dVJ_y(II,JJ,KK) + S.w1(p+1)/S.dy*(V_PS(II,JJ+p,KK)-V_PS(II,JJ-p,KK));
			dVJ_z(II,JJ,KK) = dVJ_z(II,JJ,KK) + S.w1(p+1)/S.dz*(V_PS(II,JJ,KK+p)-V_PS(II,JJ,KK-p));
			dVJ_ref_x(II,JJ,KK) = dVJ_ref_x(II,JJ,KK) + S.w1(p+1)/S.dx*(V_PS_ref(II+p,JJ,KK)-V_PS_ref(II-p,JJ,KK));
			dVJ_ref_y(II,JJ,KK) = dVJ_ref_y(II,JJ,KK) + S.w1(p+1)/S.dy*(V_PS_ref(II,JJ+p,KK)-V_PS_ref(II,JJ-p,KK));
			dVJ_ref_z(II,JJ,KK) = dVJ_ref_z(II,JJ,KK) + S.w1(p+1)/S.dz*(V_PS_ref(II,JJ,KK+p)-V_PS_ref(II,JJ,KK-p));
		end

		dVJ_1 = S.grad_T(1,1)*dVJ_x + S.grad_T(2,1)*dVJ_y + S.grad_T(3,1)*dVJ_z;
		dVJ_2 = S.grad_T(1,2)*dVJ_x + S.grad_T(2,2)*dVJ_y + S.grad_T(3,2)*dVJ_z;
		dVJ_3 = S.grad_T(1,3)*dVJ_x + S.grad_T(2,3)*dVJ_y + S.grad_T(3,3)*dVJ_z;
		dVJ_ref_1 = S.grad_T(1,1)*dVJ_ref_x + S.grad_T(2,1)*dVJ_ref_y + S.grad_T(3,1)*dVJ_ref_z;
		dVJ_ref_2 = S.grad_T(1,2)*dVJ_ref_x + S.grad_T(2,2)*dVJ_ref_y + S.grad_T(3,2)*dVJ_ref_z;
		dVJ_ref_3 = S.grad_T(1,3)*dVJ_ref_x + S.grad_T(2,3)*dVJ_ref_y + S.grad_T(3,3)*dVJ_ref_z;
		dbJ_1 = S.grad_T(1,1)*dbJ_x + S.grad_T(2,1)*dbJ_y + S.grad_T(3,1)*dbJ_z;
		dbJ_2 = S.grad_T(1,2)*dbJ_x + S.grad_T(2,2)*dbJ_y + S.grad_T(3,2)*dbJ_z;
		dbJ_3 = S.grad_T(1,3)*dbJ_x + S.grad_T(2,3)*dbJ_y + S.grad_T(3,3)*dbJ_z;
		dbJ_ref_1 = S.grad_T(1,1)*dbJ_ref_x + S.grad_T(2,1)*dbJ_ref_y + S.grad_T(3,1)*dbJ_ref_z;
		dbJ_ref_2 = S.grad_T(1,2)*dbJ_ref_x + S.grad_T(2,2)*dbJ_ref_y + S.grad_T(3,2)*dbJ_ref_z;
		dbJ_ref_3 = S.grad_T(1,3)*dbJ_ref_x + S.grad_T(2,3)*dbJ_ref_y + S.grad_T(3,3)*dbJ_ref_z;

		% Calculate local stress and correction stress components
		[II_rb,JJ_rb,KK_rb] = ndgrid(ii_s:ii_e,jj_s:jj_e,kk_s:kk_e);
		Rowcount_rb = (KK_rb-1)*S.Nx*S.Ny + (JJ_rb-1)*S.Nx + II_rb;

		[xr,yr,zr] = ndgrid((ii_s-1:ii_e-1)*S.dx - x0_i,(jj_s-1:jj_e-1)*S.dy - y0_i,(kk_s-1:kk_e-1)*S.dz - z0_i) ;
		%[xcoord,ycoord,zcoord] = meshgrid((ii_s-1:ii_e-1)*0 - x0_i,(jj_s-1:jj_e-1)*0 - y0_i,(kk_s-1:kk_e-1)*0 - z0_i) ;
		x1 = S_T(1,1)*xr + S_T(1,2)*yr + S_T(1,3)*zr;
		y1 = S_T(2,1)*xr + S_T(2,2)*yr + S_T(2,3)*zr;
		z1 = S_T(3,1)*xr + S_T(3,2)*yr + S_T(3,3)*zr;

		stress(1,1) = stress(1,1) + sum(sum(sum( dVJ_1(II,JJ,KK) .* x1 .* ( - 0.5*bJ(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(1,2) = stress(1,2) + sum(sum(sum( dVJ_1(II,JJ,KK) .* y1 .* ( - 0.5*bJ(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(1,3) = stress(1,3) + sum(sum(sum( dVJ_1(II,JJ,KK) .* z1 .* ( - 0.5*bJ(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(2,1) = stress(2,1) + sum(sum(sum( dVJ_2(II,JJ,KK) .* x1 .* ( - 0.5*bJ(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(2,2) = stress(2,2) + sum(sum(sum( dVJ_2(II,JJ,KK) .* y1 .* ( - 0.5*bJ(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(2,3) = stress(2,3) + sum(sum(sum( dVJ_2(II,JJ,KK) .* z1 .* ( - 0.5*bJ(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(3,1) = stress(3,1) + sum(sum(sum( dVJ_3(II,JJ,KK) .* x1 .* ( - 0.5*bJ(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(3,2) = stress(3,2) + sum(sum(sum( dVJ_3(II,JJ,KK) .* y1 .* ( - 0.5*bJ(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(3,3) = stress(3,3) + sum(sum(sum( dVJ_3(II,JJ,KK) .* z1 .* ( - 0.5*bJ(II,JJ,KK) ) .* S.W(Rowcount_rb) )));

		stress(1,1) = stress(1,1) + sum(sum(sum( dbJ_1(II,JJ,KK) .* x1 .* ( S.phi(Rowcount_rb) - 0.5*V_PS(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(1,2) = stress(1,2) + sum(sum(sum( dbJ_1(II,JJ,KK) .* y1 .* ( S.phi(Rowcount_rb) - 0.5*V_PS(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(1,3) = stress(1,3) + sum(sum(sum( dbJ_1(II,JJ,KK) .* z1 .* ( S.phi(Rowcount_rb) - 0.5*V_PS(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(2,1) = stress(2,1) + sum(sum(sum( dbJ_2(II,JJ,KK) .* x1 .* ( S.phi(Rowcount_rb) - 0.5*V_PS(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(2,2) = stress(2,2) + sum(sum(sum( dbJ_2(II,JJ,KK) .* y1 .* ( S.phi(Rowcount_rb) - 0.5*V_PS(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(2,3) = stress(2,3) + sum(sum(sum( dbJ_2(II,JJ,KK) .* z1 .* ( S.phi(Rowcount_rb) - 0.5*V_PS(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(3,1) = stress(3,1) + sum(sum(sum( dbJ_3(II,JJ,KK) .* x1 .* ( S.phi(Rowcount_rb) - 0.5*V_PS(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(3,2) = stress(3,2) + sum(sum(sum( dbJ_3(II,JJ,KK) .* y1 .* ( S.phi(Rowcount_rb) - 0.5*V_PS(II,JJ,KK) ) .* S.W(Rowcount_rb) )));
		stress(3,3) = stress(3,3) + sum(sum(sum( dbJ_3(II,JJ,KK) .* z1 .* ( S.phi(Rowcount_rb) - 0.5*V_PS(II,JJ,KK) ) .* S.W(Rowcount_rb) )));

		stress(1,1) = stress(1,1) + 0.5 * sum(sum(sum( ( dbJ_1(II,JJ,KK) .* ( S.V_c(Rowcount_rb) + V_PS(II,JJ,KK) ) + ...
			dbJ_ref_1(II,JJ,KK) .* ( S.V_c(Rowcount_rb) - V_PS_ref(II,JJ,KK) ) + ...
			dVJ_ref_1(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ_ref(II,JJ,KK)) - ...
			dVJ_1(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ(II,JJ,KK)) ) .* x1 .* S.W(Rowcount_rb) )));
		stress(1,2) = stress(1,2) + 0.5 * sum(sum(sum( ( dbJ_1(II,JJ,KK) .* ( S.V_c(Rowcount_rb) + V_PS(II,JJ,KK) ) + ...
			dbJ_ref_1(II,JJ,KK) .* ( S.V_c(Rowcount_rb) - V_PS_ref(II,JJ,KK) ) + ...
			dVJ_ref_1(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ_ref(II,JJ,KK)) - ...
			dVJ_1(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ(II,JJ,KK)) ) .* y1 .* S.W(Rowcount_rb) )));
		stress(1,3) = stress(1,3) + 0.5 * sum(sum(sum( ( dbJ_1(II,JJ,KK) .* ( S.V_c(Rowcount_rb) + V_PS(II,JJ,KK) ) + ...
			dbJ_ref_1(II,JJ,KK) .* ( S.V_c(Rowcount_rb) - V_PS_ref(II,JJ,KK) ) + ...
			dVJ_ref_1(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ_ref(II,JJ,KK)) - ...
			dVJ_1(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ(II,JJ,KK)) ) .* z1 .* S.W(Rowcount_rb) )));
		stress(2,1) = stress(2,1) + 0.5 * sum(sum(sum( ( dbJ_2(II,JJ,KK) .* ( S.V_c(Rowcount_rb) + V_PS(II,JJ,KK) ) + ...
			dbJ_ref_2(II,JJ,KK) .* ( S.V_c(Rowcount_rb) - V_PS_ref(II,JJ,KK) ) + ...
			dVJ_ref_2(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ_ref(II,JJ,KK)) - ...
			dVJ_2(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ(II,JJ,KK)) ) .* x1 .* S.W(Rowcount_rb) )));
		stress(2,2) = stress(2,2) + 0.5 * sum(sum(sum( ( dbJ_2(II,JJ,KK) .* ( S.V_c(Rowcount_rb) + V_PS(II,JJ,KK) ) + ...
			dbJ_ref_2(II,JJ,KK) .* ( S.V_c(Rowcount_rb) - V_PS_ref(II,JJ,KK) ) + ...
			dVJ_ref_2(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ_ref(II,JJ,KK)) - ...
			dVJ_2(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ(II,JJ,KK)) ) .* y1 .* S.W(Rowcount_rb) )));
		stress(2,3) = stress(2,3) + 0.5 * sum(sum(sum( ( dbJ_2(II,JJ,KK) .* ( S.V_c(Rowcount_rb) + V_PS(II,JJ,KK) ) + ...
			dbJ_ref_2(II,JJ,KK) .* ( S.V_c(Rowcount_rb) - V_PS_ref(II,JJ,KK) ) + ...
			dVJ_ref_2(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ_ref(II,JJ,KK)) - ...
			dVJ_2(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ(II,JJ,KK)) ) .* z1 .* S.W(Rowcount_rb) )));
		stress(3,1) = stress(3,1) + 0.5 * sum(sum(sum( ( dbJ_3(II,JJ,KK) .* ( S.V_c(Rowcount_rb) + V_PS(II,JJ,KK) ) + ...
			dbJ_ref_3(II,JJ,KK) .* ( S.V_c(Rowcount_rb) - V_PS_ref(II,JJ,KK) ) + ...
			dVJ_ref_3(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ_ref(II,JJ,KK)) - ...
			dVJ_3(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ(II,JJ,KK)) ) .* x1 .* S.W(Rowcount_rb) )));
		stress(3,2) = stress(3,2) + 0.5 * sum(sum(sum( ( dbJ_3(II,JJ,KK) .* ( S.V_c(Rowcount_rb) + V_PS(II,JJ,KK) ) + ...
			dbJ_ref_3(II,JJ,KK) .* ( S.V_c(Rowcount_rb) - V_PS_ref(II,JJ,KK) ) + ...
			dVJ_ref_3(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ_ref(II,JJ,KK)) - ...
			dVJ_3(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ(II,JJ,KK)) ) .* y1 .* S.W(Rowcount_rb) )));
		stress(3,3) = stress(3,3) + 0.5 * sum(sum(sum( ( dbJ_3(II,JJ,KK) .* ( S.V_c(Rowcount_rb) + V_PS(II,JJ,KK) ) + ...
			dbJ_ref_3(II,JJ,KK) .* ( S.V_c(Rowcount_rb) - V_PS_ref(II,JJ,KK) ) + ...
			dVJ_ref_3(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ_ref(II,JJ,KK)) - ...
			dVJ_3(II,JJ,KK) .* ( S.b(Rowcount_rb) + S.b_ref(Rowcount_rb) - bJ(II,JJ,KK)) ) .* z1 .* S.W(Rowcount_rb) )));

	end


	% Check if same type of atoms are over
	if count_typ_atms == S.Atm(count_typ).n_atm_typ
		count_typ_atms = 1;
		count_typ = count_typ + 1;
	else
		count_typ_atms = count_typ_atms + 1;
	end

end % end of loop over atoms

fprintf('\n[\b"electrostatic Stress in GPa"\n\n\n]\b');
disp((stress-stress_temp)/(S.Jacb*S.L1*S.L2*S.L3)*2.94210119*(10^4));
stress_temp = stress;

%**********************************************************************
%*                   Stress contribution from nonlocal          *
%**********************************************************************

for ks = 1:S.tnkpt*S.nspin
	if ks <= S.tnkpt
		kpt = ks;
	else
		kpt = ks - S.tnkpt;
	end

	if (kpt(1) == 0 && kpt(2) == 0 && kpt(3) == 0)
		fac = 1.0;
	else
		fac = 1.0i;
	end

	kpt_vec = S.kptgrid(kpt,:);
	
	Dpsi_x = blochGradient(S,kpt_vec,1)*S.psi(:,:,ks);
	Dpsi_y = blochGradient(S,kpt_vec,2)*S.psi(:,:,ks);
	Dpsi_z = blochGradient(S,kpt_vec,3)*S.psi(:,:,ks);
	
	TDpsi_1 = S.grad_T(1,1)*Dpsi_x + S.grad_T(2,1)*Dpsi_y + S.grad_T(3,1)*Dpsi_z ;
	TDpsi_2 = S.grad_T(1,2)*Dpsi_x + S.grad_T(2,2)*Dpsi_y + S.grad_T(3,2)*Dpsi_z ;
	TDpsi_3 = S.grad_T(1,3)*Dpsi_x + S.grad_T(2,3)*Dpsi_y + S.grad_T(3,3)*Dpsi_z ;

	for JJ_a = 1:S.n_atm % loop over all atoms
		integral_1 = zeros(S.Atom(JJ_a).angnum,S.Nev);
		integral_2_xx = zeros(S.Atom(JJ_a).angnum,S.Nev);
		integral_2_xy = zeros(S.Atom(JJ_a).angnum,S.Nev);
		integral_2_xz = zeros(S.Atom(JJ_a).angnum,S.Nev);
		integral_2_yx = zeros(S.Atom(JJ_a).angnum,S.Nev);
		integral_2_yy = zeros(S.Atom(JJ_a).angnum,S.Nev);
		integral_2_yz = zeros(S.Atom(JJ_a).angnum,S.Nev);
		integral_2_zx = zeros(S.Atom(JJ_a).angnum,S.Nev);
		integral_2_zy = zeros(S.Atom(JJ_a).angnum,S.Nev);
		integral_2_zz = zeros(S.Atom(JJ_a).angnum,S.Nev);
		
		Chi_X_mult1 = zeros(S.Atom(JJ_a).angnum,S.Nev);
		
		for img = 1:S.Atom(JJ_a).n_image_rc
			phase_fac = (exp(dot(kpt_vec,(S.Atoms(JJ_a,:)-S.Atom(JJ_a).rcImage(img).coordinates)*fac)));
			Chi_X_mult1 = Chi_X_mult1 + transpose(bsxfun(@times, conj(S.Atom(JJ_a).rcImage(img).Chi_mat), S.W(S.Atom(JJ_a).rcImage(img).rc_pos))) * S.psi(S.Atom(JJ_a).rcImage(img).rc_pos,:,ks) * phase_fac ;
		end
		
		stress(1,1) = stress(1,1) - S.occfac * S.wkpt(kpt) * transpose(S.Atom(JJ_a).gamma_Jl) * (Chi_X_mult1.*conj(Chi_X_mult1)) * S.occ(:,ks) ;
		stress(2,2) = stress(2,2) - S.occfac * S.wkpt(kpt) * transpose(S.Atom(JJ_a).gamma_Jl) * (Chi_X_mult1.*conj(Chi_X_mult1)) * S.occ(:,ks) ;
		stress(3,3) = stress(3,3) - S.occfac * S.wkpt(kpt) * transpose(S.Atom(JJ_a).gamma_Jl) * (Chi_X_mult1.*conj(Chi_X_mult1)) * S.occ(:,ks) ;
		
		for img = 1:S.Atom(JJ_a).n_image_rc
			phase_fac = (exp(dot(kpt_vec,(S.Atoms(JJ_a,:)-S.Atom(JJ_a).rcImage(img).coordinates)*fac)));
			ChiW = transpose(bsxfun(@times, conj(S.Atom(JJ_a).rcImage(img).Chi_mat), S.W(S.Atom(JJ_a).rcImage(img).rc_pos)));
			integral_1 = integral_1 + conj(ChiW) * conj(S.psi(S.Atom(JJ_a).rcImage(img).rc_pos,:,ks)) * conj(phase_fac);
			xr =(S.Atom(JJ_a).rcImage(img).rc_pos_ii-1)*S.dx - S.Atom(JJ_a).rcImage(img).coordinates(1) ;
			yr =(S.Atom(JJ_a).rcImage(img).rc_pos_jj-1)*S.dy - S.Atom(JJ_a).rcImage(img).coordinates(2) ;
			zr =(S.Atom(JJ_a).rcImage(img).rc_pos_kk-1)*S.dz - S.Atom(JJ_a).rcImage(img).coordinates(3) ;
			x_1 = S_T(1,1)*xr + S_T(1,2)*yr + S_T(1,3)*zr;
			y_1 = S_T(2,1)*xr + S_T(2,2)*yr + S_T(2,3)*zr;
			z_1 = S_T(3,1)*xr + S_T(3,2)*yr + S_T(3,3)*zr;
			
			integral_2_xx = integral_2_xx + ChiW * ...
				((TDpsi_1(S.Atom(JJ_a).rcImage(img).rc_pos,:)).*repmat(x_1,1,S.Nev)) * phase_fac ;
			
			integral_2_xy = integral_2_xy + ChiW * ...
				((TDpsi_1(S.Atom(JJ_a).rcImage(img).rc_pos,:)).*repmat(y_1,1,S.Nev)) * phase_fac ;
			
			integral_2_xz = integral_2_xz + ChiW * ...
				((TDpsi_1(S.Atom(JJ_a).rcImage(img).rc_pos,:)).*repmat(z_1,1,S.Nev)) * phase_fac ;
			
			integral_2_yx = integral_2_yx + ChiW * ...
				((TDpsi_2(S.Atom(JJ_a).rcImage(img).rc_pos,:)).*repmat(x_1,1,S.Nev)) * phase_fac ;
			
			integral_2_yy = integral_2_yy + ChiW * ...
				((TDpsi_2(S.Atom(JJ_a).rcImage(img).rc_pos,:)).*repmat(y_1,1,S.Nev)) * phase_fac ;
			
			integral_2_yz = integral_2_yz + ChiW * ...
				((TDpsi_2(S.Atom(JJ_a).rcImage(img).rc_pos,:)).*repmat(z_1,1,S.Nev)) * phase_fac ;
			
			integral_2_zx = integral_2_zx + ChiW * ...
				((TDpsi_3(S.Atom(JJ_a).rcImage(img).rc_pos,:)).*repmat(x_1,1,S.Nev)) * phase_fac ;
			
			integral_2_zy = integral_2_zy + ChiW * ...
				((TDpsi_3(S.Atom(JJ_a).rcImage(img).rc_pos,:)).*repmat(y_1,1,S.Nev)) * phase_fac ;
			
			integral_2_zz = integral_2_zz + ChiW * ...
				((TDpsi_3(S.Atom(JJ_a).rcImage(img).rc_pos,:)).*repmat(z_1,1,S.Nev)) * phase_fac ;
			
		end
		tf_xx = transpose(S.Atom(JJ_a).gamma_Jl) * real(integral_1.*integral_2_xx) * S.occ(:,ks);
		tf_xy = transpose(S.Atom(JJ_a).gamma_Jl) * real(integral_1.*integral_2_xy) * S.occ(:,ks);
		tf_xz = transpose(S.Atom(JJ_a).gamma_Jl) * real(integral_1.*integral_2_xz) * S.occ(:,ks);
		tf_yx = transpose(S.Atom(JJ_a).gamma_Jl) * real(integral_1.*integral_2_yx) * S.occ(:,ks);
		tf_yy = transpose(S.Atom(JJ_a).gamma_Jl) * real(integral_1.*integral_2_yy) * S.occ(:,ks);
		tf_yz = transpose(S.Atom(JJ_a).gamma_Jl) * real(integral_1.*integral_2_yz) * S.occ(:,ks);
		tf_zx = transpose(S.Atom(JJ_a).gamma_Jl) * real(integral_1.*integral_2_zx) * S.occ(:,ks);
		tf_zy = transpose(S.Atom(JJ_a).gamma_Jl) * real(integral_1.*integral_2_zy) * S.occ(:,ks);
		tf_zz = transpose(S.Atom(JJ_a).gamma_Jl) * real(integral_1.*integral_2_zz) * S.occ(:,ks);
		stress(1,1) = stress(1,1) - 2 * S.occfac * S.wkpt(kpt) * tf_xx;
		stress(1,2) = stress(1,2) - 2 * S.occfac * S.wkpt(kpt) * tf_xy;
		stress(1,3) = stress(1,3) - 2 * S.occfac * S.wkpt(kpt) * tf_xz;
		stress(2,1) = stress(2,1) - 2 * S.occfac * S.wkpt(kpt) * tf_yx;
		stress(2,2) = stress(2,2) - 2 * S.occfac * S.wkpt(kpt) * tf_yy;
		stress(2,3) = stress(2,3) - 2 * S.occfac * S.wkpt(kpt) * tf_yz;
		stress(3,1) = stress(3,1) - 2 * S.occfac * S.wkpt(kpt) * tf_zx;
		stress(3,2) = stress(3,2) - 2 * S.occfac * S.wkpt(kpt) * tf_zy;
		stress(3,3) = stress(3,3) - 2 * S.occfac * S.wkpt(kpt) * tf_zz;
		
	end % end of loop over atoms
end

fprintf('\n[\b"nonlocal Stress in GPa"\n\n\n]\b');
disp((stress-stress_temp)/(S.Jacb*S.L1*S.L2*S.L3)*2.94210119*(10^4));

%**********************************************************************
%*                   Stress contribution from exact exchange          *
%**********************************************************************
if S.usefock > 0
    stress_exx = zeros(3,3);
    diag_term = 0;
    
    for spin = 1:S.nspin
        spin_shift = (spin-1)*S.tnkpt;
        for k_ind = 1:S.tnkpt
            kpt_vec = S.kptgrid(k_ind,:);

            for q_ind = 1:S.tnkpthf
                % q_ind_rd is the index in reduced kptgrid
                q_ind_rd = S.kpthf_ind(q_ind,1);
                for i = 1:S.Nev
                    for j = 1:S.Nev
                        if S.kpthf_ind(q_ind,2)
                            psiqi = S.psi(:,i,q_ind_rd+spin_shift);
                        else
                            psiqi = conj(S.psi(:,i,q_ind_rd+spin_shift));
                        end
                        psikj = S.psi(:,j,k_ind+spin_shift);
                        rhs = conj(psiqi) .* psikj;

                        k = S.kptgrid(k_ind,:);
                        q = S.kptgridhf(q_ind,:);
                        k_shift = k - q;
                        [phi1, phi2] = exx_FFT_stress(S,rhs,k_shift);

                        Dphi_x = blochGradient(S,kpt_vec,1)*phi1;
                        Dphi_y = blochGradient(S,kpt_vec,2)*phi1;
                        Dphi_z = blochGradient(S,kpt_vec,3)*phi1;

                        TDphi_1 = S.grad_T(1,1)*Dphi_x + S.grad_T(2,1)*Dphi_y + S.grad_T(3,1)*Dphi_z;
                        TDphi_2 = S.grad_T(1,2)*Dphi_x + S.grad_T(2,2)*Dphi_y + S.grad_T(3,2)*Dphi_z;
                        TDphi_3 = S.grad_T(1,3)*Dphi_x + S.grad_T(2,3)*Dphi_y + S.grad_T(3,3)*Dphi_z;
        
                        Drho_x = blochGradient(S,kpt_vec,1)*rhs;
                        Drho_y = blochGradient(S,kpt_vec,2)*rhs;
                        Drho_z = blochGradient(S,kpt_vec,3)*rhs;
                        
                        TDcrho_1 = conj(S.grad_T(1,1)*Drho_x + S.grad_T(2,1)*Drho_y + S.grad_T(3,1)*Drho_z);
                        TDcrho_2 = conj(S.grad_T(1,2)*Drho_x + S.grad_T(2,2)*Drho_y + S.grad_T(3,2)*Drho_z);
                        TDcrho_3 = conj(S.grad_T(1,3)*Drho_x + S.grad_T(2,3)*Drho_y + S.grad_T(3,3)*Drho_z);

                        stress_exx(1,1) = stress_exx(1,1) - S.wkpt(k_ind)*S.wkpthf(q_ind)*S.occ_outer(i,q_ind_rd+spin_shift)*S.occ_outer(j,k_ind+spin_shift)*real(sum(S.hyb_mixing.*TDcrho_1.*TDphi_1.*S.W));
                        stress_exx(2,2) = stress_exx(2,2) - S.wkpt(k_ind)*S.wkpthf(q_ind)*S.occ_outer(i,q_ind_rd+spin_shift)*S.occ_outer(j,k_ind+spin_shift)*real(sum(S.hyb_mixing.*TDcrho_2.*TDphi_2.*S.W));
                        stress_exx(3,3) = stress_exx(3,3) - S.wkpt(k_ind)*S.wkpthf(q_ind)*S.occ_outer(i,q_ind_rd+spin_shift)*S.occ_outer(j,k_ind+spin_shift)*real(sum(S.hyb_mixing.*TDcrho_3.*TDphi_3.*S.W));
                        stress_exx(1,2) = stress_exx(1,2) - S.wkpt(k_ind)*S.wkpthf(q_ind)*S.occ_outer(i,q_ind_rd+spin_shift)*S.occ_outer(j,k_ind+spin_shift)*real(sum(S.hyb_mixing.*TDcrho_1.*TDphi_2.*S.W));
                        stress_exx(1,3) = stress_exx(1,3) - S.wkpt(k_ind)*S.wkpthf(q_ind)*S.occ_outer(i,q_ind_rd+spin_shift)*S.occ_outer(j,k_ind+spin_shift)*real(sum(S.hyb_mixing.*TDcrho_1.*TDphi_3.*S.W));
                        stress_exx(2,3) = stress_exx(2,3) - S.wkpt(k_ind)*S.wkpthf(q_ind)*S.occ_outer(i,q_ind_rd+spin_shift)*S.occ_outer(j,k_ind+spin_shift)*real(sum(S.hyb_mixing.*TDcrho_2.*TDphi_3.*S.W));
                        diag_term = diag_term - S.wkpt(k_ind)*S.wkpthf(q_ind)*S.occ_outer(i,q_ind_rd+spin_shift)*S.occ_outer(j,k_ind+spin_shift)*real(sum(S.hyb_mixing.*conj(rhs).*phi2.*S.W));
                    end
                end
            end
        end
    end
    
    stress_exx(2,1) = stress_exx(1,2);
    stress_exx(3,1) = stress_exx(1,3);
    stress_exx(3,2) = stress_exx(2,3);
    stress_exx = stress_exx/2*S.occfac;
    % convert to cartesian coordinates
    stress_exx = S.grad_T'*stress_exx*S.grad_T;
    % compute final stress_exx
    stress_exx = 2*stress_exx + (2*diag_term-2*S.Eex)*eye(3); 
    stress = stress + stress_exx;
    
    fprintf('\n[\b"exx Stress in GPa"\n\n\n]\b');
	disp(stress_exx/(S.Jacb*S.L1*S.L2*S.L3)*2.94210119*(10^4));
end

cell_measure = S.Jacb;
if S.BCx == 0
	cell_measure = cell_measure * S.L1;
end
if S.BCy == 0
	cell_measure = cell_measure * S.L2;
end
if S.BCz == 0
	cell_measure = cell_measure * S.L3;
end

stress = stress / cell_measure;
end



function [V1,V2] = exx_FFT_stress(S,rhs,k_shift)
shift_ind = find(ismembertol(S.k_shift,k_shift,1e-8,'ByRows',true))+0;
if shift_ind < S.num_shift
    u = rhs .* S.neg_phase(:,shift_ind);
else
    u = rhs;
end
u = reshape(u,S.Nx,S.Ny,S.Nz);
u_hat = fftn(u);
const_by_alpha = zeros(S.Nx,S.Ny,S.Nz);
const_by_alpha(:) = S.const_stress(shift_ind,:,:,:);
V1 = ifftn(u_hat.*const_by_alpha);
if shift_ind < S.num_shift
    V1 = V1(:) .* S.pos_phase(:,shift_ind);
else
    V1 = V1(:);
end
if S.isgamma
    V1 = real(V1(:));
end

if S.exxdivmethod == 0
    const_by_alpha(:) = S.const_stress_2(shift_ind,:,:,:);
    V2 = ifftn(u_hat.*const_by_alpha);
    if shift_ind < S.num_shift
        V2 = V2(:) .* S.pos_phase(:,shift_ind);
    else
        V2 = V2(:);
    end
    if S.isgamma
        V2 = real(V2(:));
    end
end
end





