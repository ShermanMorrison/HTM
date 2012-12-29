function [activ_col,P,boost,inhmat,overlap_hist,activ_hist] =...
    Space_Pool(feedforward,synmat,P,pthresh,minoverlap,boost,...
    neighbormat,inhmat,overlap_hist,activ_hist,stepnum)

% Space_Pool inputs:
% feedforward:  feedforward input
% synmat:       potential synapse matrix between (input,col)
% P:            permanence matrix corresponding to synmat
% pthresh:      permanence threshold for synapse to be established
% minoverlap:   threshold for feedforward activation of column
% boost:        boost vector for the inhibition battle between columns
% neighbormat:  fixed matrix where (i,j) == 1  ==> col j can inhibit col i
% inhmat:       changing matrix where (i,j) == 1 ==> col j inhibits col i
% overlap_hist: # of times each column has been activated
% activ_hist:   # of times each column has won inhibition battle
% stepnum:      current iteration #

% Space_Pool outputs:
% activ_col:    active columns
% P:            updated permanence matrix between columns and input
% boost:        updated boost vector
% overlap_hist: updated overlap_hist
% activ_hist:   updated activ_hist



[numIn,numCol] = size(synmat);

pot_syn = synmat; % fixed matrix of potential synapses
est_syn = 1 * (P >= pthresh); % active synapse connectivity matrix


%% For each step, update active col, permanences, boost, & inhibition rad.

% Determine overlap

overlap = zeros(numCol,1);

for i = 1:numCol
    overlap(i) = dot(est_syn(:,i),feedforward);
end


% activation =  boost * overlap

% columns with low overlap become inactive
overlap = overlap .* (overlap >= minoverlap); 
% activity of above-threshold columns
activ_str = boost .* overlap; 


% Inhibit all the less active neighbors

num_activ = 1; % set number of active neighbors for each column

% 1 for activ_col's, 0 for inactive col's
activ_col = 1*(activ_str > 0); 
% columns with above-threshold feed-forward activation
survivor_col = zeros(numCol); 


for i = 1:numCol
    % inactivate all columns not in the top active group for all landlord
    % columns
    weedout = activ_col; 
    
    neighb_index = find(inhmat(:,i)>0); % index of column i's neighbors
    neighb_activ_str = activ_str(neighb_index); % activ_str of neighbors
    
    % sort neighbor column activity descending
    ranked_activ_str = sort(neighb_activ_str,'descend'); 
    
    
    % Choose rank "num_activ" as min_activity, otherwise choose min
    % active column
    if length(ranked_activ_str) >= num_activ
        min_activity = ranked_activ_str(num_activ);
    else
        min_activity = min(ranked_activ_str); % if
    end
    
    for j = 1:length(neighb_index)
        if activ_str(neighb_index(j)) < min_activity
            weedout(neighb_index(j)) = 0; % weedout neighbors with less 
            % than threshold min_activity for those neighbors, 
            % the rest remain active
        end
    end
    survivor_col(:,i) = weedout; % each neighbor chooses
    % num_activ neighbors to let live. neighbors don't kill themselves. 
    % columns chosen to live by all neighbors survive
    
end

% column i, entry j is whether or not HTM column i survived the inhibition
% by HTM col j.
survivors = prod(survivor_col,2); % surviving active columns


% Update Permanence Matrix

activ_col_index = find(survivors > 0);

for i = 1:length(activ_col_index)
    
    % logical index for potential synpases to active col. i
    pot_syn_index = pot_syn(:,activ_col_index(i)) > 0; 
    
    % activ_syn =  active input && potential synapse
    activ_syn_index =  (pot_syn_index .* feedforward) > 0; 
    % the other synapses are inactive
    inactiv_syn_index = (pot_syn_index - activ_syn_index) > 0; 
    
    % increment perm of active syn
    P(activ_syn_index,activ_col_index(i)) = ...
        min( P(activ_syn_index,activ_col_index(i)) + .05, 1); 
    
    % decrement perm of inactive syn
    P(inactiv_syn_index,activ_col_index(i)) = ...
        max( P(inactiv_syn_index,activ_col_index(i)) - .05, 0); 
    
end

% Update boost vector

overlap_hist = overlap_hist + activ_col;
overlap_ave = 1/stepnum * overlap_hist;

activ_hist = activ_hist + survivors;
activ_ave = 1/stepnum * activ_hist;

minduty = 1/4 * max(activ_ave);


for i = 1:numCol
    
    % increase permanences for under-activated columns
    if overlap_ave(i) < minduty
        P(:,i) = min(1.01 * P(:,i),1);
    end
    
    % increase boost factor for under-active columns
    if activ_ave(i) < minduty
        boost(i) = boost(i) + 0.1;
    else
        boost(i) = 1;
    end
    
end

% Update inhibition radius

% active synapse connectivity matrix
est_syn = 1 * (P > pthresh); 

% new inhrad is ave # of synapse connections of the columns
inhrad = min(round(numIn*mean(mean(est_syn))),numCol - 1); 
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




