%% run_HTM

clear all

%% Create synmat, initial P, pthresh, boost, inhrad, & Input
% synmat: feed-forward connectivity
% P: permanence matrix
% boost: factor boosting active column activity
% Input: Input at each stepnum

% synmat(i,j) = 1 ==> col j has a potential synapse with Input i
% here we have 400 columns each receiving a subset of 400 Inputs

synmat = randi([0 14], 400, 400);
synmat = ones(400,400) .* (synmat==1);

synmat2 = randi([0 14], 400, 400);
synmat2 = ones(400,400) .* (synmat2==1);
[numIn, numCol] = size(synmat);

P = .3 * synmat; % permanence matrix, must be initialized > pthresh
P2 = .3 * synmat;

pthresh = 0.25; % threshold for established synapse

boost = ones(numCol,1); % boost vector for inhibition battle
boost2 = ones(numCol,1);

minoverlap = 3; % threshold for activation

% Each column keeps the desired number of neighbor columns, ie the most
% active ones
% Each column starts off inhibiting the [# inhrad] neighbors below its
% diagonal
%  1 =< inhrad =< numCol - 1

neighbormat = ones(numCol) - eye(numCol);
neighbormat2 = neighbormat;

inhrad = round(numCol-1);
inhrad2 = inhrad;

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

inhmat2 = inhmat;

% total times each column has been activated
overlap_hist = zeros(numCol,1);
overlap_hist2 = overlap_hist;
% total times each column has been active
activ_hist = zeros(numCol,1);
activ_hist2 = activ_hist;

%% Input


% SEQUENCE

A = round(numIn/100*[2 6 10 14 18]);

B = round(numIn/100*[82 86 90 94 98]);

C = [-5 -5 -5 -5 -5];

In_Seq = {A A B A B A B A B A B B B A A B B A A B B A B C C A A A A A B B B B B};

totstepnum = 175;

Input = zeros(numIn,totstepnum);

count = 0;

for i = 1:length(In_Seq)
    [~,ins] = size(In_Seq{i});
    for j = 1:ins
        count = count + 1;
        for k = 1:length(In_Seq{i}(:,j))
            low_bound = max(In_Seq{i}(k,j)-5,1);
            high_bound = min(In_Seq{i}(k,j)+5,numIn);
            Input(low_bound:high_bound,count) = 1;
        end
    end
end

% Make 4th sequence Z = (A union B)

% for j = 16:20
%     Input(:,j) = Input(:,j) + Input(:,j-5);
% end


%% Run Space-Time


activ_col_hist = zeros(numCol,totstepnum);
predictiv_col_hist = activ_col_hist;
Input_hist = zeros(numIn,totstepnum);

activ_col_hist2 = activ_col_hist;
predictiv_col_hist2 = activ_col_hist;
Input_hist2 = Input_hist;

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

active_cells2 = active_cells;
predictive_cells2 = predictive_cells;
LS2 = LS;
distdend2 = distdend;
P_lat2 = P_lat;
P_lat_prev2 = P_lat_prev;
segUpdateMap2 = segUpdateMap;
segUpdateList2 = segUpdateList;
activ_seg_syn2 = activ_seg_syn;
activ_seg_index2 = activ_seg_index;

for stepnum = 1:totstepnum
    
    for region = 1:totlayers
        
        if region == 1
            feedforward = Input(:,stepnum);
        else
            feedforward = col_output;
        end
        
        % Region 1
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
        
        
        % Region 2
        feedforward_cells = 1*((active_cells + predictive_cells) > 0);
        feedforward2 = floor((find(feedforward_cells)-1)/3)+1;
        feedforward2 = intersect((1:numCol)',feedforward2);
        
        a = zeros(numIn,1);
        a(feedforward2) = 1;
        feedforward2 = a;
        
        [activ_col2,P2,boost2,inhmat2,overlap_hist2,activ_hist2] = ...
            Space_Pool(feedforward2,synmat2,P2,pthresh,minoverlap,...
            boost2,neighbormat2,inhmat2,overlap_hist2,activ_hist2,stepnum);
        
        p2 = floor((find(predictive_cells2)-1)/3) + 1;
        p2 = intersect((1:numCol)',p2);
        predictiv_col_hist2(p2,stepnum) = 1;
        activ_col_hist2(:,stepnum) = activ_col2;
        Input_hist2(:,stepnum) = feedforward2;
        
        [active_cells2,predictive_cells2,LS2,distdend2,P_lat2,...
            P_lat_prev2,activ_seg_syn2,activ_seg_index2,...
            segUpdateMap2,segUpdateList2] = ...
            Time_Pool(activ_col2,active_cells2,predictive_cells2,...
            LS2,distdend2,P_lat2,P_lat_prev2,segUpdateMap2,segUpdateList2,...
            activ_seg_syn2,activ_seg_index2);
        
    end
    
end

% Region 1
% prediction in t-1 is predicted active state in t

correct_pred = activ_col_hist.*predictiv_col_hist;

figure(1)
spy(correct_pred,10,'g')
hold on
unanticipated = activ_col_hist - correct_pred;
spy(unanticipated,10,'r')

overpredicted = predictiv_col_hist - correct_pred;
% spy(overpredicted,10,'k')

title('Region 1: Anticipated Activity (Green), Unexpected Activity (Red)')
xlabel('Iteration #')
ylabel('Column #')

% Region 2
% prediction in t-1 is predicted active state in t

correct_pred2 = activ_col_hist2.*predictiv_col_hist2;

figure(2)
spy(correct_pred2,10,'g')
hold on
unanticipated2 = activ_col_hist2 - correct_pred2;
spy(unanticipated2,10,'r')


overpredicted2 = predictiv_col_hist2 - correct_pred2;
% spy(overpredicted,10,'k')

title('Region 2: Anticipated Activity (Green), Unexpected Activity (Red)')
xlabel('Iteration #')
ylabel('Column #')






