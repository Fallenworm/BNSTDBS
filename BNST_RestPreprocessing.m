
%% load raw data
clc;clear;
LoadPath = ['D:\',  'Rest_EEG_analysis\BNST_data', '\RawData\'];
SavePath = ['D:\',  'Rest_EEG_analysis\BNST_data', '\FilteredData\'];
File = dir(fullfile(LoadPath,'*.vhdr'));
Subj = {File.name};

%% filtering 
for  m = 1 : size(Subj,2)
    EEG = pop_loadbv(LoadPath, Subj{1,m}, [], []);
    EEG = eeg_checkset(EEG);

    %% Bipolar offline
    for j = 1:7
        L_LFP = EEG.data(j+1,:) - EEG.data(j,:);
        EEG.data(j,:,:) = L_LFP;
    end
    for k = 9:15
        R_LFP = EEG.data(k+1,:) - EEG.data(k,:);
        EEG.data(k,:,:) = R_LFP;
    end
    EEG = pop_select(EEG,'channel',{'L0' 'L1' 'L2' 'L3' 'L4' 'L5' 'L6' 'R0' 'R1' 'R2' 'R3' 'R4' 'R5' ...
        'R6' 'FP1' 'FP2' 'Fz' 'F3' 'F4' 'F7' 'F8' 'HEOG' 'uVEOG' 'lVEOG' 'ECG'});

    %% Filter
    EEG = pop_eegfiltnew(EEG, 'locutoff',48,'hicutoff',52,'revfilt',1);
    EEG  = pop_basicfilter( EEG,  1:25 , 'Boundary', 'boundary', 'Cutoff',  1, 'Design', 'butter', 'Filter', 'highpass', 'Order',  2, 'RemoveDC', 'on' );

    %% Save datasets
    pop_saveset(EEG, 'filename',Subj{1,m},'filepath',SavePath);
    clear EEG
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ICA
clc;clear;
LoadPath =  ['D:\',  'Rest_EEG_analysis\BNST_data', '\FilteredData\'];
SavePath =  ['D:\',  'Rest_EEG_analysis\BNST_data', '\SegData\'];
File = dir(fullfile(LoadPath,'*.set'));
Subj = {File.name};
for  m = 1 : size(Subj,2)
    EEG = pop_loadset('filename',Subj{1, m},'filepath',LoadPath);
    EEG = eeg_checkset( EEG );
    EEG = pop_resample( EEG, 500);
    EEG.event = [];
    EEG = eeg_regepochs(EEG, 'recurrence', 2, 'limits',[0 2], 'rmbase',NaN);
    EEG=pop_chanedit(EEG, 'lookup','D:\\Toolbox\\eeglab2022.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc');
    EEG = pop_saveset(EEG, 'filename',Subj{1,m},'filepath',SavePath);
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input Parameters
clc;clear;
LoadPath =  ['D:\', 'Rest_EEG_analysis\BNST_data', '\FinalDatafixed\'];
File = dir(fullfile(LoadPath,'*.set'));
Subj = {File.name}';
% load('Demograph.mat');
for m = 1:size(Subj,1)
    fprintf(['Enter channels of ', Subj{m,1},'\n']) ;
    chanIDs = unique(UI_cellArray(1, {}), 'stable') ;
    Subj{m,4} = chanIDs;
end
BNSTdata(:,4) = Subj(:,4);
save('D:\Rest_EEG_analysis\BNST_data\BNSTdata','BNSTdata');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Merge epoched data
clc;clear;
LoadPath =  ['D:\', 'Rest_EEG_analysis\BNST_data', '\RejData\'];
SavePath =  ['D:\', 'Rest_EEG_analysis\BNST_data', '\FinalData1\'];
File = dir(fullfile(LoadPath,'*.set'));
Subj = {File.name};
for  m = 1 : size(Subj,2)
    m =13;
    EEG = pop_loadset('filename',Subj{1,m},'filepath',LoadPath);
    for i = 1:length(EEG.epoch)
        ALLEEG(i) = pop_selectevent( EEG, 'epoch',i,'deleteevents','off','deleteepochs','on','invertepochs','off'); 
    end
    EEG = pop_mergeset( ALLEEG, 1:length(EEG.epoch), 0); 
    EEG.event = []; 
    EEG = pop_saveset(EEG, 'filename',Subj{1,m},'filepath',SavePath);
end