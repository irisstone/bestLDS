%% set data info

folder = 'datasetB';
dataset = 'datasetB_large';

%% Import Data

data = load(sprintf('data/model-comparisons/folder/%s.mat',folder,dataset));

% true params
A = data.A;
B = data.B;
C = data.C;
D = data.D;
Q = data.Q;

log_trues = zeros(5,1);
log_bestlds = zeros(5,1);
log_ems = zeros(5,1);


%% run loop
for i = 1:5

%%data
ss = transpose(squeeze(data.u_test(i,:,:)));
yy = transpose(squeeze(data.y_test(i,:,:)));
ntrials = data.N/5;
nz = size(data.A,2);

% Compute posterior mean given true parameters
[zzmap_true,H,logevtrue] = computeMAP_LDSBernoulli_wInputs(zeros(nz,ntrials),yy,ss,A,B,C,D,Q);
fprintf('log-evidence given true params: %.2f\n', logevtrue);

log_trues(i) = logevtrue;

%%best-lds parameters

params = load(sprintf('data/model-comparisons/folder/%s-best-lds-fit-%d.mat',folder,dataset,i));

% best-lds params
A = params.Ahat;
B = params.Bhat;
C = params.Chat;
D = params.Dhat;
Q = data.Q;

% Compute posterior mean given true parameters
[zzmap_bestlds,H,logevbestlds] = computeMAP_LDSBernoulli_wInputs(zeros(nz,ntrials),yy,ss,A,B,C,D,Q);
fprintf('log-evidence given best-lds params: %.2f\n', logevbestlds);

log_bestlds(i) = logevbestlds;

end

avg_log_true = mean(log_trues);
avg_log_bestlds = mean(log_bestlds);

fprintf('average log-evidence given true params: %.2f\n', avg_log_true);
fprintf('average log-evidence given best lds params: %.2f\n', avg_log_bestlds);