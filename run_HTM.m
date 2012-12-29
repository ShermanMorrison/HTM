%% run_HTM

clear all

%% Create synmat, initial P, pthresh, boost, inhrad, & Input
% synmat: feed-forward connectivity
% P: permanence matrix
% boost: factor boosting active column activity
% Input: Input at each stepnum

% synmat(i,j) = 1 ==> col j has a potential synapse with Input i
% here we have 200 columns each receiving a subset of 400 Inputs

synmat = randi([0 14], 400, 400);
synmat = ones(400,400) .* (synmat==1);
[numIn, numCol] = size(synmat);

P = .3 * synmat; % permanence matrix, must be initialized > pthresh
pthresh = 0.25; % threshold for established synapse

boost = ones(numCol,1); % boost vector for inhibition battle

minoverlap = 3; % threshold for activation

% Each column keeps the desired number of neighbor columns, ie the most
% active ones
% Each column starts off inhibiting the [# inhrad] neighbors below its
% diagonal
%  1 =< inhrad =< numCol - 1

neighbormat = ones(numCol) - eye(numCol);

inhrad = round(numCol-1);

% Inhibit first [inhrad] neighbors
inhmat = neighbormat;

for i = 1:numCol
    if  i + inhrad > numCol
        start = i + inhrad - numCol + 1;
        inhmat(start:i,i) = 0;
    else
        inhmat(i+inhrad+1:end, i) = 0;
        inhmat(1:i,i) = 0;
    end
end

% total times each column has been activated
overlap_hist = zeros(numCol,1);
% total times each column has been active
activ_hist = zeros(numCol,1);

%% Input

% OUTSIDE TEMPERATURE

w = 2*pi/24;
t = (1:1:24*4);



Temp = 81 + 7*sin(w*t(1:end/2));
Temp = [Temp 81 + 0*sin(w*t(1+end/2:3*end/4))];
Temp = [Temp 81 + 7*sin(w*t(1+3*end/4:end))];
  

totstepnum = length(t);

Input = zeros(numIn,totstepnum);

In_Temp = round(numIn*(1 + (Temp - 81)/7)/2);

for i = 1:length(t)
    low_bound = max(In_Temp(i) - 5,1);
    high_bound = min(In_Temp(i) + 5,numIn);
    
    Input(low_bound:high_bound,i) = 1;
end



% CYCLE
% totstepnum = 100; % iterations of incoming Input
% 
% Input = zeros(numIn,totstepnum); % Input columns
% Input(:,1) = zeros(numIn,1); % first Input
% Input(1:5,1) = 1;
% 
% % transformation from Input_k to Input_k+1
% T = eye(numIn);
% T(:,[1,numIn]) = T(:,[numIn,1]);
% 
% % get all Input columns
% 
% for stepnum = 2:totstepnum
%     
%     Input(:,stepnum) = T*Input(:,stepnum-1);
%     
% end

%% Run Space-Time


activ_col_hist = zeros(numCol,totstepnum);
predictiv_col_hist = activ_col_hist;
Input_hist = zeros(numIn,totstepnum);

totlayers = 1;

cells_per_col = 3;

active_cells = zeros(cells_per_col,numCol);
predictive_cells = zeros(cells_per_col,numCol);
LS = zeros(cells_per_col,numCol);
distdend = cell(cells_per_col,numCol);
P_lat = cell(cells_per_col,numCol);
P_lat_prev = cell(cells_per_col,numCol);
segUpdateMap = [];
segUpdateList = cell(cells_per_col,numCol);
activ_seg_syn = {};
activ_seg_index = [];

for stepnum = 1:totstepnum
    
    for region = 1:totlayers
        
        if region == 1
            feedforward = Input(:,stepnum);
        else
            feedforward = col_output;
        end
        
        [activ_col,P,boost,inhmat,overlap_hist,activ_hist] = ...
            Space_Pool(feedforward,synmat,P,pthresh,minoverlap,...
            boost,neighbormat,inhmat,overlap_hist,activ_hist,stepnum);
        
        
        p = floor((find(predictive_cells)-1)/3) + 1;
        p = intersect((1:numCol)',p);
        predictiv_col_hist(p,stepnum) = 1;
        activ_col_hist(:,stepnum) = activ_col;
        Input_hist(:,stepnum) = feedforward;
        
        
        [active_cells,predictive_cells,LS,distdend,P_lat,...
            P_lat_prev,activ_seg_syn,activ_seg_index,...
            segUpdateMap,segUpdateList] = ...
            Time_Pool(activ_col,active_cells,predictive_cells,...
            LS,distdend,P_lat,P_lat_prev,segUpdateMap,segUpdateList,...
            activ_seg_syn,activ_seg_index);
         
%           display(stepnum)
        
          
    end
    
end

% prediction in t-1 is predicted active state in t
predicted_col_hist = [zeros(numCol,1), predictiv_col_hist];
predicted_col_hist(:,end) = [];

correct_pred = activ_col_hist.*predictiv_col_hist;

spy(correct_pred,10,'g')

hold on

unanticipated = activ_col_hist - correct_pred;
spy(unanticipated,10,'r')


overpredicted = predictiv_col_hist - correct_pred;
% spy(overpredicted,10,'k')


title('Anticipated Activity (Green), Unexpected Activity (Red)')
xlabel('Iteration #')
ylabel('Column #')






