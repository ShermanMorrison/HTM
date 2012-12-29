function [active_cells,predictive_cells,LS,distdend,P_lat,P_lat_prev,...
    activ_seg_syn,activ_seg_index,segUpdateMap,segUpdateList] = ...
    Time_Pool(activ_col,active_cells_prev,predictive_cells_prev,...
    LS_prev,distdend,P_lat,P_lat_prev,segUpdateMap,segUpdateList,...
    activ_seg_syn_prev,activ_seg_index_prev)

% Time_Pool inputs:
% activ_col: list of active columns from feedforward input
% active_cells_prev: list of active cells in t-1
% predictive_cells_prev: list of predictive cells in t-1
% LS_prev: matrix of cell learning states in t-1
% distdend = {}; % t-1 connectedness
% P_lat: t permanences
% P_lat_prev: t-1 permanences for weakly active seqsegs/ for non-seq
% segs
%
% segUpdateMap:  row = [neuron, col, seg #, (addsyn)/(seqseg)]
% leftover instructions for reenforcing segment / adding segment to cell
% from t-1
%
% segUpdateList: {neuron,col}{i} = {{seg#} {[ , ] ... [ , ]}} = ith seg to
% update and its synapses.
% leftover list of synapses on each seg to reenforce from t-1
%
% activ_seg_syn_prev = { }; % {i} = {activ syn# ... activ syn#} = active
% synapses in t-1 corresponding to segment in row i of activ_seg_index_prev
%
% activ_seg_index_prev: row = [neuron, col, seqseg, seg#]
% active segments in t-1

% Time_Pool outputs:
% active_cells: list of active cells in t after Phase 1 FF activation
% predictive_cells: list of predictive cells in t after Phase 2 lateral
% activation
% LS: matrix of cell learning states in t
% distdend: t/t+1 connectedness  before/after Phase 3
% P_lat: t/t+1 permanences before/after Phase 3
% P_lat_prev: t-1/t permanences before/after Phase 3
%
% activ_seg_syn = { }; % {i} = {activ syn# ... activ syn#} = active
% synapses in t corresponding to segment in row i of activ_seg_index_prev
%
% activ_seg_index: row = [neuron, col, seqseg, seg#]
% active segments in t
%
% segUpdateMap:  row = [neuron, col, seg #, (addsyn)/(seqseg)]
% updated instructions for reenforcing segment / adding segment to cell
%
% segUpdateList: {neuron,col}{i} = {{seg#} {[ , ] ... [ , ]}} = ith seg to
% update and its synapses.
% updated list of synapses on each seg to reenforce




%% Initialize
numCol = 400; % # active col ~40
cells_per_col = 3;
predict_thresh = 4;
min_thresh = 1;
newSynCount = 5;

connected_perm = .25;
initial_perm = .3;
inc_perm = .05;
dec_perm = .04;

% new states
active_cells = zeros(cells_per_col,numCol);
predictive_cells = zeros(cells_per_col,numCol);
LS = zeros(cells_per_col,numCol);
activ_seg_syn = {};
activ_seg_index = [];


%% Phase 1
% For each active col, activate predictive cells & choose a learn cell and
% seq seg

activ_col_index = find(activ_col == 1);

for i = 1:length(activ_col_index)
    col = activ_col_index(i);
    
    for neuron = 1:cells_per_col
        
        predicted = 0;
        learncell = 0;
        
        % if cell is predictive
        if predictive_cells_prev(neuron,col) == 1
            
            % if cell had active seq. seg. in t-1, activate cell
            
            was_pred_by_seq_seg = strmatch([neuron,col,1],...
                activ_seg_index_prev(:,1:3));
            
            was_predicted = length(was_pred_by_seq_seg)>= 1;
            
            % if cell had activ seq seg
            if was_predicted
                % activate cell
                active_cells(neuron,col) = 1;
                % mark column activity as predicted
                predicted = predicted + 1;
                
                % choose cell as learn cell if predicted by learn cells
                % for each active seq. seg to cell, get numsyn from learn
                % cells
                for j = 1:length(was_pred_by_seq_seg)
                    
                    num_learn_syn = 0;
                    counter = was_pred_by_seq_seg(j);
                    
                    numsyn = length(activ_seg_syn_prev{counter});
                    
                    % check if active syn from learn cells >=predict_thresh
                    for syn = 1:numsyn
                        
                        syn_cell = activ_seg_syn_prev{counter}{syn}(1);
                        syn_col = activ_seg_syn_prev{counter}{syn}(2);
                        
                        if LS_prev(syn_cell,syn_col) == 1
                            num_learn_syn = num_learn_syn + 1;
                        end
                    end
                    
                    % if num_learn_syn >= predict_thresh, seq seg is
                    % making cell learn. mark cell as learn cell.
                    % mark segmentUpdateMap to strengthen these seq segs.
                    
                    if num_learn_syn >= predict_thresh
                        LS(neuron,col) = 1;
                    end
                end
            end
        end
    end
    
    % if col activity unpredicted, activate all cells
    if predicted == 0
        active_cells(:,col) = 1;
    end
    
    % if no learn cell chosen in active col, get most active seqseg in t-1
    if learncell == 0
        % for each cell with numseg >= 1
        
        best_cell = 0;
        best_cell_seg = 0;
        best_cell_seg_activation = 0;
        
        for neuron = 1:cells_per_col
            
            if isempty(P_lat_prev{neuron,col})
                break
            end
            
            numseg =  length(P_lat_prev{neuron,col});
            if numseg >= 1
                
                seg_active_count = zeros(1,numseg);
                
                % for each seg with numsyn >= min_thresh
                for seg = 1:numseg
                    numsyn = length(P_lat_prev{neuron,col}{seg});
                    
                    % is seg is seq seg
                    if distdend{neuron,col}{seg}{1}{1}
                    
                        if numsyn >= min_thresh
                            % count number of active syn
                            for syn = 1:numsyn
                                
                                syn_cell = distdend{neuron,col}{seg}{2}{syn}(1);
                                syn_col = distdend{neuron,col}{seg}{2}{syn}(2);
                                
                                % if syn is to active cell in t-1 via
                                % established syn in t-1
                                
                                syn_active = ...
                                    active_cells_prev(syn_cell,syn_col) == 1;
                                syn_est = P_lat_prev{neuron,col}{seg}(syn) >= ...
                                    connected_perm;
                                
                                if syn_active && syn_est
                                    seg_active_count(seg) = ...
                                        seg_active_count(seg) + 1;
                                end
                            end
                        end
                    end
                    
                end
                
                
                % get best_seg and corresponding activation of the cell
                [best_seg_activation,best_seg] = max(seg_active_count);
                
                % if best_seg activation is the highest of all cells, make
                % corresponding seg = best_cell_seg
                % corresponding cell = best_cell
                if best_seg_activation > best_cell_seg_activation
                    
                    best_cell = neuron;
                    best_cell_seg = best_seg;
                    best_cell_seg_activation = best_seg_activation;
                end
            end
        end
        
        % if best_cell_seg has activation >= min_thresh,
        % mark segUpdateMap to enforce it & getSegActiveSynapses on list
        % mark best_cell as learn cell
        if best_cell_seg_activation >= min_thresh
            
            segUpdateMap = [segUpdateMap; ...
                [best_cell, col, best_cell_seg, 1]];
            
            % get active syn
            
            numsyn = length(P_lat_prev{best_cell,col}{best_cell_seg});
            segList = {};
            for syn = 1:numsyn
                
                syn_cell = distdend{best_cell,col}{best_cell_seg}{2}{syn}(1);
                syn_col = distdend{best_cell,col}{best_cell_seg}{2}{syn}(2);
                
                % if syn is to active cell in t-1 via
                % established syn in t-1
                
                syn_active = active_cells_prev(syn_cell,syn_col) == 1;
                
                syn_est = P_lat_prev{best_cell,col}{best_cell_seg}(syn) >=...
                    connected_perm;
                
                if syn_active && syn_est
                    segList = [segList {[syn_cell,syn_col]}];
                end
            end
            
            segUpdateList{best_cell,col} = [segUpdateList{best_cell,col};...
                {best_cell_seg} {segList}];
            
            LS(best_cell,col) = 1;
            
        else
            % choose cell with least segments and add seq seg
            
            if isempty([distdend{:,col}])
                cell_least_seg = randi(cells_per_col);
            else
                
                cell_segnum = zeros(1,cells_per_col);
                
                for neuron = 1:cells_per_col
                    cell_segnum(neuron) = length(distdend{neuron,col});
                end
                
                [~,cell_least_seg] = min(cell_segnum);
                
                
            end
            
            segUpdateMap = [segUpdateMap; [cell_least_seg col -1 1]];
            LS(cell_least_seg,col) = 1;
            
        end
        
    end
end

%% Phase 2
% Mark active segs to be strengthened.
% Put cells with active segs in predictive state.
% For each predictive cell, mark best non-seq seg from t-1 to be
% strengthened.

for col = 1:numCol
    for neuron = 1:cells_per_col
        numseg = length(distdend{neuron,col});
        
        % mark each active seg for strengthening,
        % corresponding cell = predictive
        
        for seg = 1:numseg
            numsyn = length(distdend{neuron,col}{seg}{2});
            
            seg_activation = 0;
            
            % count active syn to the seg
            for syn = 1:numsyn
                
                syn_cell = distdend{neuron,col}{seg}{2}{syn}(1);
                syn_col = distdend{neuron,col}{seg}{2}{syn}(2);
                
                syn_active = active_cells(syn_cell,syn_col);
                
                syn_est = P_lat{neuron,col}{seg}(syn) >= connected_perm;
                
                if syn_active && syn_est
                    seg_activation = seg_activation + 1;
                end
            end
            
            % if seg is activated, cell = predictive
            % record seg in activ_seg_syn & activ_seg_index
            % mark seg for strengthening in segUpdateMap
            % getSegmentActiveSynapses in segUpdateList
            
            if seg_activation >= predict_thresh
                predictive_cells(neuron,col) = 1;
                
                seqseg = distdend{neuron,col}{seg}{1}{1};
                
                activ_seg_index = [activ_seg_index;...
                    [neuron,col,seqseg,seg]];
                
                segUpdateMap = [segUpdateMap; [neuron col seg 0]];
                
                % get active syn
                numsyn = length(P_lat{neuron,col}{seg});
                segList = {};
                for syn = 1:numsyn
                    
                    syn_cell = distdend{neuron,col}{seg}{2}{syn}(1);
                    syn_col = distdend{neuron,col}{seg}{2}{syn}(2);
                    
                    % if syn is to active cell via
                    % established syn
                    
                    syn_active = active_cells(syn_cell,syn_col) == 1;
                    syn_est = P_lat{neuron,col}{seg}(syn) >=connected_perm;
                    
                    if syn_active && syn_est
                        segList = [segList {[syn_cell,syn_col]}];
                    end
                end
                
                activ_seg_syn = [activ_seg_syn; {segList}];
                segUpdateList{neuron,col} = [segUpdateList{neuron,col};...
                    {seg} {segList}];
                
            end
        end
        
        % if cell is now predictive, strengthen best non-seq seg from t-1
        if predictive_cells(neuron,col) == 1
            numseg = length(P_lat_prev{neuron,col});
            
            seg_activation = zeros(1,numseg);
            % for each non-seq seg
            for seg = 1:numseg
                if distdend{neuron,col}{seg}{1}{1} == 0
                    
                    % count active syn to seg in t-1
                    numsyn = length(P_lat_prev{neuron,col}{seg});
                    for syn = 1:numsyn
                        
                        syn_cell = distdend{neuron,col}{seg}{2}{syn}(1);
                        syn_col = distdend{neuron,col}{seg}{2}{syn}(2);
                        
                        syn_active = ...
                            active_cells_prev(syn_cell,syn_col) == 1;
                        syn_est = P_lat_prev{neuron,col}{seg}(syn) >=...
                            connected_perm;
                        
                        if syn_active && syn_est
                            seg_activation(seg) = seg_activation(seg) + 1;
                        end
                    end
                end
            end
            
            [best_seg_activation, best_seg] = max(seg_activation);
            
            % if best_seg_activation >= min_thresh, mark segUpdateMap to
            % reenforce this segment and add synapses
            % getSegmentActiveSynapses in segUpdateList
            
            if best_seg_activation >= min_thresh
                segUpdateMap = [segUpdateMap; [neuron, col, best_seg, 1]];
                
                % get active syn
                numsyn = length(P_lat_prev{neuron,col}{best_seg});
                segList = {};
                for syn = 1:numsyn
                    
                    syn_cell = distdend{neuron,col}{best_seg}{2}{syn}(1);
                    syn_col = distdend{neuron,col}{best_seg}{2}{syn}(2);
                    
                    % if syn is to active cell in t-1 via
                    % established syn in t-1
                    
                    syn_active = active_cells_prev(syn_cell,syn_col) == 1;
                    syn_est = P_lat_prev{neuron,col}{best_seg}(syn) >=...
                        connected_perm;
                    
                    if syn_active && syn_est
                        segList = [segList {[syn_cell,syn_col]}];
                    end
                end
                
                segUpdateList{neuron,col} = [segUpdateList{neuron,col};...
                    {best_seg} {segList}];
                
            else
                % add non-seq segment to the predictive cell
                segUpdateMap = [segUpdateMap; [neuron col -1 0]];
            end
            
        end
    end
end


%% Phase 3
% Implement queued segment updates:
% positive reenforcement to learn cells,
% negative reenforcement to non-learn cells which stop predicting

% save P_lat used in this timestep as P_lat_prev for next timestep
P_lat_prev = P_lat;

for col = 1:numCol
    for neuron = 1:cells_per_col
        
        % if cell is in learn state
        if LS(neuron,col) == 1
            
            % get all queued updates for the cell
            index = strmatch([neuron,col],segUpdateMap(:,1:2));
            
            % each update corresponds to a segment
            for j = 1:length(index)
                
                % get update location on segUpdateMap
                counter = index(j);
                
                % if seg marked for reenforcement, positively reenforce
                if segUpdateMap(counter,3) > 0
                    
                    seg = segUpdateMap(counter,3);
                    
                    % get all potential syn to seg
                    numsyn = length(distdend{neuron,col}{seg}{2});
                    all_syn = [distdend{neuron,col}{seg}{2}{:}];
                    all_syn = reshape(all_syn,2,numsyn)';
                    
                    % get index of all active syn to seg
                    update_index = ...
                        find([segUpdateList{neuron,col}{:,1}] == seg);
                    
                    % positively reenforce all queued updates to this seg
                    for i = 1:length(update_index)
                        
                        update_num = update_index(i);
                        
                        num_activ_syn = ...
                            length(segUpdateList{neuron,col}{update_num,2});
                        
                        % positively reenforce active segments by inc_perm
                        for syn = 1:num_activ_syn
                            
                            syn_cell = ...
                                segUpdateList{neuron,col}...
                                {update_num,2}{syn}(1);
                            syn_col = ...
                                segUpdateList{neuron,col}...
                                {update_num,2}{syn}(2);
                            
                            syn_to_strengthen = ...
                                strmatch([syn_cell,syn_col],all_syn);
                            
                            P_lat{neuron,col}{seg}(syn_to_strengthen) =...
                                P_lat{neuron,col}{seg}(syn_to_strengthen)+...
                                inc_perm + dec_perm;
                        end
                        
                        % weaken remaining segments by dec_perm
                        P_lat{neuron,col}{seg}(:) = ...
                            P_lat{neuron,col}{seg}(:) - dec_perm;
                    end
                    
                    for i = 1:length(update_index)
                        
                        update_num = update_index(i);
                        
                        % delete the segUpdateList row
                        segUpdateList{neuron,col}{update_num,1} = [];
                        segUpdateList{neuron,col}{update_num,2} = [];
                        
                        segUpdateList{neuron,col} = ...
                            segUpdateList{neuron,col}(~cellfun(...
                            'isempty',segUpdateList{neuron,col}));
                        num_remaining_updates = ...
                            length(segUpdateList{neuron,col})/2;
                        segUpdateList{neuron,col} = ...
                            reshape(segUpdateList{neuron,col},...
                            num_remaining_updates,2);
                        
                    end
                    
                    
                    % add (newSynCount - existing synapses to learn cells)
                    % learn synapses if marked
                    if segUpdateMap(counter,4) == 1
                        
                        [learn_cell_index,learn_col_index] = find(LS);
                        learn_cells = [learn_cell_index, learn_col_index];
                        
                        [rows_LC,~] = size(learn_cells);
                        LC_index = (1:rows_LC)';
                        
                        all_syn_cells = [distdend{neuron,col}{seg}{2}{:}];
                        all_syn_cells = reshape(all_syn_cells,2,numsyn)';
                        
                        [~,existing_LC_index] = ...
                            intersect(learn_cells,all_syn_cells,'rows');
                        
                        if length(existing_LC_index) >= newSynCount
                            break
                        else
                            
                            numsyn_to_add = ...
                                min(newSynCount - length(existing_LC_index),...
                                length(LC_index) - length(existing_LC_index));
                            
                            % get LC_index not already in existing syn
                            LC_index(existing_LC_index) = 0;
                            LC_index = find(LC_index);
                            
                            order = randperm(length(LC_index));
                            
                            LC_to_add_index = LC_index(order(1:numsyn_to_add));
                        end
                        
                        % add each LC synapse
                        for i = 1:length(LC_to_add_index)
                            
                            syn = LC_to_add_index(i);
                            
                            
                            
                            new_syn_cell = learn_cells(syn,1);
                            new_syn_col = learn_cells(syn,2);
                            
                            
                            % add the synapse
                            distdend{neuron,col}{seg}{2} = ...
                                [ distdend{neuron,col}{seg}{2} ...
                                {[new_syn_cell new_syn_col]} ];
                            
                            % initialize synapse with initial_perm
                            P_lat{neuron,col}{seg} =...
                                [P_lat{neuron,col}{seg} initial_perm];
                            
                            
                        end
                    end
                    
                    % add segment to learn cell if marked
                    % initialize segment with up to newSynCount learn cells
                else
                    
                    % mark new_seg as seq or non-seq
                    if segUpdateMap(counter,4) == 1
                        new_seg = {{1} {}};
                    else
                        new_seg = {{0} {}};
                    end
                    
                    % initialize segment with as many learn cells in
                    % LS_prev as possible up to newSynCount
                    
                    % if there were no previous learn cells, add empty seg
                    if isempty(find(LS_prev, 1))
                        
                        % add new_seg to distdend
                        distdend{neuron,col} = [ distdend{neuron,col}...
                            {new_seg} ];
                        % add new_seg permanences to P_lat
                        P_lat{neuron,col} = [P_lat{neuron,col} {[]}];
                        
                    else
                        % add as many existing prev LC up to newSynCount
                        
                        [learn_cell_index,learn_col_index] = find(LS_prev);
                        learn_cells = [learn_cell_index, learn_col_index];
                        
                        [rows_LC,~] = size(learn_cells);
                        
                        num_newLC = min(rows_LC,newSynCount);
                        
                        LC_index = randperm(rows_LC)';
                        
                        LC_index = LC_index(1:num_newLC,:);
                        
                        LC_to_add = learn_cells(LC_index,:);
                        
                        
                        % add synapses to new_seg
                        for i = 1:num_newLC
                            new_seg{2} = [new_seg{2} ...
                                [LC_to_add(i,1) LC_to_add(i,2)] ];
                        end
                        
                        % initialize synapse perms
                        init_syn_perm = initial_perm*ones(1,num_newLC);
                        
                        % add new_seg to distdend
                        distdend{neuron,col} = [ distdend{neuron,col}...
                            {new_seg} ];
                        % add new_seg permanences to P_lat
                        P_lat{neuron,col} = [ P_lat{neuron,col} ...
                            {init_syn_perm} ];
                        
                    end
                    
                end
            end
            
            for j = 1:length(index)
                
                % get update location on segUpdateMap
                counter = index(j);
                
                % delete the segUpdateMap row
                segUpdateMap(counter,:) = 0;
                segUpdateMap = segUpdateMap(any(segUpdateMap,2),:);
                
            end
            
            % if cell is not learning and stops predicting from t-1
            
        elseif not(predictive_cells(neuron,col)) && ...
                predictive_cells_prev(neuron,col)
            
            % get all queued updates for the cell
            index = strmatch([neuron,col],segUpdateMap(:,1:2));
            
            % each update corresponds to a segment
            for j = 1:length(index)
                
                % get update location on segUpdateMap
                counter = index(j);
                
                % if seg marked for reenforcement, negatively reenforce
                if segUpdateMap(counter,3) > 0
                    
                    seg = segUpdateMap(counter,3);
                    
                    % get all potential syn to seg
                    numsyn = length(distdend{neuron,col}{seg}{2});
                    all_syn = [distdend{neuron,col}{seg}{2}{:}];
                    all_syn = reshape(all_syn,2,numsyn)';
                    
                    % get index of all active syn to seg
                    update_index = find([segUpdateList{neuron,col}{:,1}]==...
                        seg);
                    
                    % negatively reenforce all queued updates to this seg
                    for i = 1:length(update_index)
                        
                        update_num = update_index(i);
                        
                        num_activ_syn = ...
                            length(segUpdateList{neuron,col}{update_num,2});
                        
                        % negatively reenforce active segments by inc_perm
                        for syn = 1:num_activ_syn
                            
                            syn_cell = ...
                                segUpdateList{neuron,col}...
                                {update_num,2}{syn}(1);
                            syn_col = ...
                                segUpdateList{neuron,col}...
                                {update_num,2}{syn}(2);
                            
                            syn_to_strengthen = ...
                                strmatch([syn_cell,syn_col],all_syn);
                            
                            P_lat{neuron,col}{seg}(syn_to_strengthen) = ...
                                P_lat{neuron,col}{seg}(syn_to_strengthen)-...
                                dec_perm;
                        end
                    end
                    
                    for i = 1:length(update_index)
                        
                        update_num = update_index(i);
                        
                        % delete the segUpdateList row
                        
                        segUpdateList{neuron,col}{update_num,1} = [];
                        segUpdateList{neuron,col}{update_num,2} = [];
                        
                        segUpdateList{neuron,col} = ...
                            segUpdateList{neuron,col}(~cellfun(...
                            'isempty',segUpdateList{neuron,col}));
                        num_remaining_updates = ...
                            length(segUpdateList{neuron,col})/2;
                        segUpdateList{neuron,col} = ...
                            reshape(segUpdateList{neuron,col},...
                            num_remaining_updates,2);
                        
                    end
                    
                    % add (newSynCount - existing synapses to learn cells)
                    % learn synapses if marked
                    if segUpdateMap(counter,4) == 1
                        
                        [learn_cell_index,learn_col_index] = find(LS);
                        learn_cells = [learn_cell_index, learn_col_index];
                        
                        [rows_LC,~] = size(learn_cells);
                        LC_index = (1:rows_LC)';
                        
                        all_syn_cells = [distdend{neuron,col}{seg}{2}{:}];
                        all_syn_cells = reshape(all_syn_cells,2,numsyn)';
                        
                        [~,existing_LC_index] = ...
                            intersect(learn_cells,all_syn_cells,'rows');
                        
                        if length(existing_LC_index) >= newSynCount
                            break
                        else
                            
                            numsyn_to_add = ...
                                min(newSynCount - length(existing_LC_index),...
                                length(LC_index) - length(existing_LC_index));
                            
                            % get LC_index not already in existing syn
                            LC_index(existing_LC_index) = 0;
                            LC_index = find(LC_index);
                            
                            order = randperm(length(LC_index));
                            
                            LC_to_add_index = LC_index(order(1:numsyn_to_add));
                        end
                        
                        % add each LC synapse
                        for i = 1:length(LC_to_add_index)
                            
                            syn = LC_to_add_index(i);
                            
                            
                            
                            new_syn_cell = learn_cells(syn,1);
                            new_syn_col = learn_cells(syn,2);
                            
                            
                            % add the synapse
                            distdend{neuron,col}{seg}{2} = ...
                                [ distdend{neuron,col}{seg}{2} ...
                                {[new_syn_cell new_syn_col]} ];
                            
                            % initialize synapse with initial_perm
                            P_lat{neuron,col}{seg} =...
                                [P_lat{neuron,col}{seg} initial_perm];
                            
                            
                        end
                    end
                    
                    % add segment to learn cell if marked
                    % initialize segment with up to newSynCount learn cells
                else
                    
                    % mark new_seg as seq or non-seq
                    if segUpdateMap(counter,4) == 1
                        new_seg = {{1} {}};
                    else
                        new_seg = {{0} {}};
                    end
                    
                    % initialize segment with as many learn cells in
                    % LS_prev as possible up to newSynCount
                    
                    % if there were no previous learn cells, add empty seg
                    if isempty(find(LS_prev, 1))
                        
                        % add new_seg to distdend
                        distdend{neuron,col} = [ distdend{neuron,col}...
                            {new_seg} ];
                        % add new_seg permanences to P_lat
                        P_lat{neuron,col} = [P_lat{neuron,col} {[]}];
                        
                    else
                        % add as many existing prev LC up to newSynCount
                        
                        [learn_cell_index,learn_col_index] = find(LS_prev);
                        learn_cells = [learn_cell_index, learn_col_index];
                        
                        [rows_LC,~] = size(learn_cells);
                        
                        num_newLC = min(rows_LC,newSynCount);
                        
                        LC_index = randperm(rows_LC)';
                        
                        LC_index = LC_index(1:num_newLC,:);
                        
                        LC_to_add = learn_cells(LC_index,:);
                        
                        
                        % add synapses to new_seg
                        for i = 1:num_newLC
                            new_seg{2} = [new_seg{2} ...
                                [LC_to_add(i,1) LC_to_add(i,2)] ];
                        end
                        
                        % initialize synapse perms
                        init_syn_perm = initial_perm*ones(1,num_newLC);
                        
                        % add new_seg to distdend
                        distdend{neuron,col} = [ distdend{neuron,col}...
                            {new_seg} ];
                        % add new_seg permanences to P_lat
                        P_lat{neuron,col} = [ P_lat{neuron,col} ...
                            {init_syn_perm} ];
                        
                    end
                    
                end
            end
            
            for j = 1:length(index)
                
                % get update location on segUpdateMap
                counter = index(j);
                
                % delete the segUpdateMap row
                segUpdateMap(counter,:) = 0;
                segUpdateMap = segUpdateMap(any(segUpdateMap,2),:);
                
            end
        end
    end
end






















