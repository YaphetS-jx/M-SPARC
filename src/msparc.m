function S = msparc(fname,varargin)
% @brief    M-SPARC implements DFT calculation 
%
% @param fname  The input filename.
% @param parallel_switch (optional)     1: turn on parallelization
%                                       0: turn off parallelization
%
% @authors  Qimen Xu <qimenxu@gatech.edu>
%           Abhiraj Sharma <asharma424@gatech.edu>
%           Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
%
% @copyright (c) 2019 Material Physics & Mechanics Group, Georgia Tech
%

format long;
fprintf('\n');

% Start timer
total_time = tic;

% check input arguments
if isempty(fname)
	error('Please provide input file name (excluding extension)!');
end

parallel_switch = 0; % default is off
if nargin == 2
	parallel_switch = varargin{1};
elseif nargin > 2
	error('Too many input arguments.');
end

% Read initial data and create a structure S to store the data
S = initialization(fname);

S.ofdft = 1;
S.ofdft_lambda = 0.2;
S.ofdft_Cf = 0.3*((3*pi*pi)^(2/3));

S.parallel = parallel_switch; %0 - N0 && 1 - Yes
if S.parallel == 1
	% Set up the parallel environment
	warning('off','MATLAB:maxNumCompThreads:Deprecated');
	max_threads = 12; % Experiment with this
	max_threads_default = maxNumCompThreads;
	if(max_threads > max_threads_default)
		max_threads = max_threads_default;
	end
	LASTN_all_comp = maxNumCompThreads(max_threads);
	fprintf('\n \n Starting the Matlab pool ... \n');
	tic_pool = tic;
	num_worker_heuristic =  S.tnkpt;

	% Get the default size as a safeguard
	myCluster = parcluster();
	if(myCluster.NumWorkers < num_worker_heuristic)
		num_worker_heuristic = myCluster.NumWorkers;
	end
	
	% Clean up older pools
	delete(gcp('nocreate'));
	
	% Launch new pool
	poolobj = parpool(num_worker_heuristic) ;
	S.num_worker_heuristic = num_worker_heuristic;
	fprintf('\n \n Pool set up in %f s. \n', toc(tic_pool));
end

% Perform relaxation/MD/Single point calculation
if S.RelaxFlag
	S = relax(S);
elseif S.MDFlag
	S = md(S);
else
	[~,~,S] = electronicGroundStateAtomicForce(S.Atoms,S);
end

% Print results (Final atom positions)
fprintf(' Final atomic positions (Cartesian) are as follows:\n');
for k = 1:S.n_atm
	fprintf(' %9.6f \t %9.6f \t %9.6f \n', S.Atoms(k,1),S.Atoms(k,2),S.Atoms(k,3));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if S.parallel == 1
	fprintf('\n \n Closing the Matlab pool ... \n');
	tic_pool = tic;
	delete(poolobj);
	fprintf('\n \n Pool closed in %f s. \n', toc(tic_pool));
end

% write to output file
outfname = S.outfname;
fileID = fopen(outfname,'a');
if (fileID == -1) 
	error('\n Cannot open file "%s"\n',outfname);
end 

t_wall = toc(total_time);
fprintf(fileID,'***************************************************************************\n');
fprintf(fileID,'                               Timing info                                 \n');
fprintf(fileID,'***************************************************************************\n');
fprintf(fileID,'Total walltime                     :  %.3f sec\n', t_wall);
fprintf(fileID,'___________________________________________________________________________\n');
fprintf(fileID,'\n');
fprintf(fileID,'***************************************************************************\n');
fprintf(fileID,'*             Material Physics & Mechanics Group, Georgia Tech            *\n');
fprintf(fileID,'*                       PI: Phanish Suryanarayana                         *\n');
fprintf(fileID,'*                Main Developers: Qimen Xu, Abhiraj Sharma                *\n');
fprintf(fileID,'*     Acknowledgements: U.S. DOE (DE-SC0019410); NSF (1333500,1553212)    *\n');
fprintf(fileID,'***************************************************************************\n');
fprintf(fileID,'                                                                           \n');
fclose(fileID);

% Program run-time
fprintf('\n Run-time of the program: %f seconds\n', t_wall);

if S.parallel == 1
   LASTN_reset = maxNumCompThreads(LASTN_all_comp);
end
