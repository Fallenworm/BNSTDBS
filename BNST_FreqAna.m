eeglab;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NAc_Freq_Analysis
clc;clear;
LoadPath =  ['/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data', '/Finaldatafixed/'];
File = dir(fullfile(LoadPath,'*.set'));
Subj = {File.name};
load('/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/BNSTdata.mat');

for m = 1:size(Subj,2)
    EEG = pop_loadset('filename',Subj{1,m},'filepath',LoadPath);
    if any (m == [5, 8, 10, 13, 16, 17])
        BNSTdata{m,20} = nan;
    elseif size(BNSTdata{m,2},2) == 1
        chanEEG = pop_select(EEG,'channel',BNSTdata{m,2});
        [pow_wel,Freq_wel] = pwelch(mean(chanEEG.data,1),512,0,512,500);
        BNSTdata(m,20) = mat2cell(pow_wel',1);
    elseif size(BNSTdata{m,2},2) == 2
        chanEEG = pop_select(EEG,'channel',{BNSTdata{m,2}{1,1},BNSTdata{m,2}{1,2}});
        [pow_wel,Freq_wel] = pwelch(mean(chanEEG.data,1),512,0,512,500);
        BNSTdata(m,20) = mat2cell(pow_wel',1);
    end
    clear pow_wel chanEEG EEG
end
for m = 1:size(Subj,2)
    BNSTdata{m,19} = mat2cell(Freq_wel',1);
end

save('D:/Rest_EEG_analysis/BNST_Data/BNSTdata','BNSTdata');

%% BNST_Freq_Analysis
clc;clear;
LoadPath =  ['/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data', '/Finaldatafixed/'];
File = dir(fullfile(LoadPath,'*.set'));
Subj = {File.name};
load('/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/BNSTdata.mat');

for m = 1:size(Subj,2)
    EEG = pop_loadset('filename',Subj{1,m},'filepath',LoadPath);
    if size(BNSTdata{m,3},2) == 1
        chanEEG = pop_select(EEG,'channel',BNSTdata{m,3});
        [pow_wel,Freq_wel] = pwelch(mean(chanEEG.data,1),512,0,512,500);
        BNSTdata(m,21) = mat2cell(pow_wel',1);
    elseif size(BNSTdata{m,3},2) == 2
        chanEEG = pop_select(EEG,'channel',{BNSTdata{m,3}{1,1},BNSTdata{m,3}{1,2}});
        [pow_wel,Freq_wel] = pwelch(mean(chanEEG.data,1),512,0,512,500);
        BNSTdata(m,21) = mat2cell(pow_wel',1);
    end
    clear pow_wel chanEEG EEG
end
save('D:/Rest_EEG_analysis/BNST_Data/BNSTdata','BNSTdata');

%% EEG_Freq_Analysis
clc;clear;
LoadPath =  ['/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data', '/Finaldatafixed/'];
File = dir(fullfile(LoadPath,'*.set'));
Subj = {File.name};
load('/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/BNSTdata.mat');

for m = 1:size(Subj,2)
    EEG = pop_loadset('filename',Subj{1,m},'filepath',LoadPath);
    EEG = pop_select(EEG,'channel',{'FP1' 'FP2' 'F4' 'F3' 'F7' 'F8' 'Fz'});
    [pow_wel,Freq_wel] = pwelch(squeeze(mean(EEG.data,1)),512,0,512,500);
    BNSTdata(m,25) = mat2cell(pow_wel',1); % eeg power spectrum data
    clear pow_wel EEG
end
save('D:/Rest_EEG_analysis/BNST_Data/BNSTdata','BNSTdata');


%% HT_Freq_Analysis
clc;clear;
LoadPath =  ['/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data', '/Finaldatafixed/'];
File = dir(fullfile(LoadPath,'*.set'));
Subj = {File.name};
load('/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/BNSTdata.mat');

for m = 1:size(Subj,2)
    EEG = pop_loadset('filename',Subj{1,m},'filepath',LoadPath);
    if any (m == 9)
        BNSTdata{m,22} = nan;
    elseif size(BNSTdata{m,4},2) == 1
        chanEEG = pop_select(EEG,'channel',BNSTdata{m,4});
        [pow_wel,Freq_wel] = pwelch(mean(chanEEG.data,1),512,0,512,500);
        BNSTdata(m,22) = mat2cell(pow_wel',1);
    elseif size(BNSTdata{m,4},2) == 2
        chanEEG = pop_select(EEG,'channel',{BNSTdata{m,4}{1,1},BNSTdata{m,4}{1,2}});
        [pow_wel,Freq_wel] = pwelch(mean(chanEEG.data,1),512,0,512,500);
        BNSTdata(m,22) = mat2cell(pow_wel',1);
    end
    clear pow_wel chanEEG EEG
end
save('D:/Rest_EEG_analysis/BNST_Data/BNSTdata','BNSTdata');


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Coherence
clc;clear;
LoadPath =  '/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/Finaldatafixed/';
File = dir(fullfile(LoadPath,'*.set'));
Subj = {File.name};
load('/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/BNSTdataA.mat');

for m = 1:size(Subj,2)
    EEG = pop_loadset(Subj{1,m},'/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/Finaldatafixed/');
    if any (m == [5 8 10 13 16 17])
        BNSTdata{m,25} = nan;
    else
    % define bnst data
    if size(BNSTdata{m,3},2) == 1
        BNST_LFP = pop_select(EEG,'channel',BNSTdata{m,3});
    elseif size(BNSTdata{m,3},2) == 2
        BNST_LFP = pop_select(EEG,'channel',{BNSTdata{m,3}{1,1},BNSTdata{m,3}{1,2}});
    end
    % define ht data
    if size(BNSTdata{m,2},2) == 1
        HT_LFP = pop_select(EEG,'channel',BNSTdata{m,2});
    elseif size(BNSTdata{m,2},2) == 2
        HT_LFP = pop_select(EEG,'channel',{BNSTdata{m,2}{1,1},BNSTdata{m,2}{1,2}});
    end

    Frontal_EEG = pop_select(EEG,'channel',{'FP1' 'FP2' 'F4' 'F3' 'F7' 'F8' 'Fz'});

    BNST_LFP.data = squeeze(mean(BNST_LFP.data, 1));
    HT_LFP.data = squeeze(mean(HT_LFP.data, 1));
    Frontal_EEG.data = squeeze(mean(Frontal_EEG.data,1));

    % set the sliding window and step size
    winSize = 10*500; % 10 second * 50 sample rate
    dataLength = size(BNST_LFP.data,2);
    stepvector = 1 : 0.5*winSize : dataLength; % 50% overlap
    for a = 1: size(stepvector,2)
        if (stepvector(a)+winSize-1) > dataLength
            break;
        end
        Wins(a,:) = stepvector(a):(stepvector(a)+winSize-1);
    end

    % calculate the coherence
    for i = 1:size(Wins,1)
        [cxy,fc] = mscohere(BNST_LFP.data(1,Wins(i,:)),HT_LFP.data(1,Wins(i,:)),hann(512),256,[],500);
        Cxy(i,:) = cxy;
    end
    Cxy = squeeze(mean(Cxy,1))*100;
    BNSTdata(m,25) = mat2cell(Cxy,1); % coherence
    clear Wins EEG
end
end
save('/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/BNSTDataA.mat','BNSTdata');


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Granger causality
clc;clear;
LoadPath =  '/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/Finaldatafixed/';
File = dir(fullfile(LoadPath,'*.set'));
Subj = {File.name};
load('/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/BNSTdataA.mat');

% Parameters
regmode = "LWR";
icregmode = "LWR";
morder = "BIC";
amaxlags = 2000;
tstat = "F";
alpha = 0.05;
mhtc = "FDR";
fres = [];
fs = 500;
seed = 0;
nsamps = 100;
bsize = [];
nperms = 100;

% full cycle for GC estimation
for m = 1:size(Subj,2)
    EEG = pop_loadset(Subj{1,m},'/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/Finaldatafixed/');
        % define bnst data
        if size(BNSTdata{m,3},2) == 1
            BNST_LFP = pop_select(EEG,'channel',BNSTdata{m,3});
        elseif size(BNSTdata{m,3},2) == 2
            BNST_LFP = pop_select(EEG,'channel',{BNSTdata{m,3}{1,1},BNSTdata{m,3}{1,2}});
        end

        % define eeg data
        Frontal_EEG = pop_select(EEG,'channel',{'FP1' 'FP2' 'F4' 'F3' 'F7' 'F8' 'Fz'});

        BNST_LFP.data = squeeze(mean(BNST_LFP.data, 1));
        Frontal_EEG.data = squeeze(mean(Frontal_EEG.data,1));
        % set the sliding window and step size
        winSize = 4*500; % 4 second * 50 sample rate
        dataLength = size(BNST_LFP.data,2);
        stepvector = 1 : 0.5*winSize : dataLength; % 50% overlap
        for a = 1: size(stepvector,2)
            if (stepvector(a)+winSize-1) > dataLength
                break;
            end
            Wins(a,:) = stepvector(a):(stepvector(a)+winSize-1);
        end
        for ii = 1:size(Wins,1)
            dat(:,:,ii) = [BNST_LFP.data(1,Wins(ii,:));Frontal_EEG.data(1,Wins(ii,:))];
        end

        ntrials = size(dat,3);

        % model order estimation
        [AIC, BIC, moAIC, moBIC] = tsdata_to_infocrit(dat,20,regmode);
        if strcmp(morder,"AIC")
            morder = moAIC;
        elseif strcmp(morder, "BIC")
            morder = moBIC;
        end 

        rng(seed);

        % VAR model estimation
        [A, SIG] = tsdata_to_var(dat,morder,regmode);
        assert(~isbad(A),"VAR estimation failed");

        Inf = var_info(A,SIG);
        assert(~Inf.error,"VAR error(s) found - bailing out")

        % Autocovariance calculation
        [G,info] = var_to_autocov(A, SIG, amaxlags);
        var_acinfo(info, true);

        % time domain GC calculation
        [F,pval] = var_to_pwcgc(A,SIG,dat, regmode, tstat);
        assert(~isbad(F,false),"GC estimation failed");
        sig = significance(pval,0.05,mhtc);        
        % frequency domain GC calculation
        if isempty(fres)
            fres = 2^nextpow2(Inf.acdec);
        end
        if fres > 20000
            istr = input(" ", "s"); if isempty(istr) || ~strcmpi(istr, "y"); fprintf(2, "Aborting...\n"); return; end
        end
        f = var_to_spwcgc(A, SIG,fres);
        assert(~isbad(f,false),"spectral GC estimation failed ~bailing out");

        Time_GC(m,:,:) = F;
        Freq_GC(m,:,:,:) = f;
        Siga(m,:,:) = sig;
        clear winSize EEG G F f stepvector Wins
   %end
end
for m = 1:size(Subj,2)
    BNSTdata(m,27) = mat2cell(Time_GC(m,:,:),1); % GC
    BNSTdata(m,28) = mat2cell(Freq_GC(m,:,:,:),1);
end


save('/Users/wanglinbin/Documents/LFPProject/Rest_EEG_analysis/BNST_Data/BNSTDataA.mat','BNSTdata');

clear BNSTdata

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Calculate Modulation Index (MI)
% clc;clear;
% LoadPath =  ['D:/', 'Rest_EEG_analysis/BNST_Data', '/Finaldatafixed/'];
% File = dir(fullfile(LoadPath,'*.set'));
% Subj = {File.name};
% load('D:/Rest_EEG_analysis/BNST_Data/BNSTdata.mat');
%
% %% Define the amplitude- and phase-frequencies
%
% PhaseFreqVector=4:1:12;
% AmpFreqVector=12:4:90;
%
% PhaseFreq_BandWidth=2;
% AmpFreq_BandWidth=10;
%
% srate = 500;
%
% %% Define phase bins
% nbin = 18; % number of phase bins
% position=zeros(1,nbin); % this variable will get the beginning (not the center) of each phase bin (in rads)
% winsize = 2*pi/nbin;
% for j=1:nbin
%     position(j) = -pi+(j-1)*winsize;
% end
%
% for m = 1:size(Subj,2)
%     EEG = pop_loadset('filename',Subj{1,m},'filepath',LoadPath);
%     %     if any (m == 9)
%     %         BNSTdata{m,8} = nan;
%     %    else
%     % define bnst data
%     if size(BNSTdata{m,2},2) == 1
%         BNST_LFP = pop_select(EEG,'channel',BNSTdata{m,2});
%     elseif size(BNSTdata{m,2},2) == 2
%         BNST_LFP = pop_select(EEG,'channel',{BNSTdata{m,2}{1,1},BNSTdata{m,2}{1,2}});
%     end
%     % define bnst data1
%     if size(BNSTdata{m,2},2) == 1
%         HT_LFP = pop_select(EEG,'channel',BNSTdata{m,2});
%     elseif size(BNSTdata{m,2},2) == 2
%         HT_LFP = pop_select(EEG,'channel',{BNSTdata{m,2}{1,1},BNSTdata{m,2}{1,2}});
%     end
%
%     %     % define ht data
%     %     if size(BNSTdata{m,3},2) == 1
%     %         HT_LFP = pop_select(EEG,'channel',BNSTdata{m,3});
%     %     elseif size(BNSTdata{m,3},2) == 2
%     %         HT_LFP = pop_select(EEG,'channel',{BNSTdata{m,3}{1,1},BNSTdata{m,3}{1,2}});
%     %     end
%
%     BNST_LFP.data = squeeze(mean(BNST_LFP.data, 1));
%     HT_LFP.data = squeeze(mean(HT_LFP.data, 1));
%
%     % set the sliding window and step size
%     winSize = 30*500; % 30 second * 50 sample rate
%     dataLength = size(EEG.data,2);
%     stepvector = 1 : 0.5*winSize : dataLength; % 50% overlap
%     for a = 1: size(stepvector,2)
%         if (stepvector(a)+winSize-1) > dataLength
%             break;
%         end
%         Wins(a,:) = stepvector(a):(stepvector(a)+winSize-1);
%     end
%
%     % set a empty matrix for saving data
%     Comodulogram=single(zeros(size(Wins,1),length(PhaseFreqVector),length(AmpFreqVector)));
%
%     for i = 1:size(Wins,1)
%
%         AmpFreqTransformed = zeros(length(AmpFreqVector), winSize);
%         PhaseFreqTransformed = zeros(length(PhaseFreqVector), winSize);
%
%         %% hilbert transform (need eegfilt function)
%         for ii=1:length(AmpFreqVector)
%             Af1 = AmpFreqVector(ii);
%             Af2=Af1+AmpFreq_BandWidth;
%             AmpFreq=eegfilt(BNST_LFP.data(1,Wins(i,:)),srate,Af1,Af2); % filtering
%             AmpFreqTransformed(ii, :) = abs(hilbert(AmpFreq)); % getting the amplitude envelope
%         end
%
%         for jj=1:length(PhaseFreqVector)
%             Pf1 = PhaseFreqVector(jj);
%             Pf2 = Pf1 + PhaseFreq_BandWidth;
%             PhaseFreq=eegfilt(HT_LFP.data(1,Wins(i,:)),srate,Pf1,Pf2); % filtering
%             PhaseFreqTransformed(jj, :) = angle(hilbert(PhaseFreq)); % getting the phase time series
%         end
%
%         %% Caculate MI (need Modlindex_v2 function)
%         counter1=0;
%         for ii=1:length(PhaseFreqVector)
%             counter1=counter1+1;
%
%             Pf1 = PhaseFreqVector(ii);
%             Pf2 = Pf1+PhaseFreq_BandWidth;
%
%             counter2=0;
%             for jj=1:length(AmpFreqVector)
%                 counter2=counter2+1;
%
%                 Af1 = AmpFreqVector(jj);
%                 Af2 = Af1+AmpFreq_BandWidth;
%                 [MI,MeanAmp]=ModIndex_v2(PhaseFreqTransformed(ii, :), AmpFreqTransformed(jj, :), position);
%                 Comodulogram(i,counter1,counter2)=MI;
%             end
%         end
%     end
%
%     PACdata = squeeze(mean(Comodulogram,1));
%     BNSTdata(m,33) = mat2cell(PACdata,size(PACdata,1)); % MI
%     clear Wins EEG AmpFreqTransformed PhaseFreqTransformed
% end
% % end
% save('D:/Rest_EEG_analysis/BNST_data/BNSTdata','BNSTdata');
%
%
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Calculate PLV
% clc;clear;
% LoadPath =  ['D:/', 'Rest_EEG_analysis/BNST_Data', '/Finaldatafixed/'];
% File = dir(fullfile(LoadPath,'*.set'));
% Subj = {File.name};
% load('D:/Rest_EEG_analysis/BNST_Data/BNSTdata.mat');
% band = inputdlg('the limits of band');%指定所要分析的频率的范围（单位是Hz）
% band = str2num(band{1}); %将band变量由字符转换为数值
%
% for m = 1:size(Subj,2)
%     EEG = pop_loadset('filename',Subj{1,m},'filepath',LoadPath);
%     %     if any (m == 9)
%     %         plv(m) = nan;
%     %     else
%     if size(BNSTdata{m,2},2) == 1
%         BNST_LFP = pop_select(EEG,'channel',BNSTdata{m,2});
%     elseif size(BNSTdata{m,2},2) == 2
%         BNST_LFP = pop_select(EEG,'channel',{BNSTdata{m,2}{1,1},BNSTdata{m,2}{1,2}});
%     end
%     % define HT data
%     if size(BNSTdata{m,3},2) == 1
%         HT_LFP = pop_select(EEG,'channel',BNSTdata{m,3});
%     elseif size(BNSTdata{m,3},2) == 2
%         HT_LFP = pop_select(EEG,'channel',{BNSTdata{m,3}{1,1},BNSTdata{m,3}{1,2}});
%     end
%     % define EEG data
%     EEG = pop_select(EEG,'channel',{'FP1' 'FP2' 'F4' 'F3' 'F7' 'F8' 'Fz'});
%
%
%
%     %% 提取EEG在band频段的相位
%     eeg_filtered = eegfilt(BNST_LFP.data,...
%         500,band(1,1),band(1,2),0,3*fix(500/band(1,1)),0,'fir1',0);
%     bandPhase_eeg = angle(hilbert(eeg_filtered)); %逐个分段进行Hilbert变换，并提取相位
%
%     %去掉数据中前10%和后10%的结果 因为hilbert变换对该区域不准确
%     perc10w1 =  floor(size(bandPhase_eeg,2)*0.1);% 确定数据长度10%是多少个样本点
%     bandPhase_eeg = bandPhase_eeg(:,perc10w1+1:end-perc10w1); %因Hilbert变换对数据首尾相位估算不准确，顾去掉前10%和后10%样本点的相位
%     % epoch for bandphase
%     % epoch_num1 = floor(size(band_phase,2)/size(EEG.data,2)); % 确定剩余的样本点如果转换为分段数据，可以分成多少段
%     %     band_phase = band_phase(:,1:epoch_num1*size(EEG.data,2)); % 依据可以分成的段数，截取数据
%     %     band_phase = reshape(band_phase,[size(EEG.data,1) size(EEG.data,2) epoch_num1]);% 将数据重新转换为二维：样本点*分段
%     %% 提取LFP在band频段的相位
%     lfp_filtered = eegfilt(EEG.data,...
%         500,band(1,1),band(1,2),0,3*fix(500/band(1,1)),0,'fir1',0);
%     bandPhase_LFP = angle(hilbert(lfp_filtered)); %逐个分段进行Hilbert变换，并提取相位
%
%     %去掉数据中前10%和后10%的结果 因为hilbert变换对该区域不准确
%     perc10w2 =  floor(size(bandPhase_LFP,2)*0.1);% 确定数据长度10%是多少个样本点
%     bandPhase_LFP = bandPhase_LFP(:,perc10w2+1:end-perc10w2); %因Hilbert变换对数据首尾相位估算不准确，顾去掉前10%和后10%样本点的相位
%     % epoch for bandphase
%     %     epoch_num2 = floor(size(band_phase_LFP,2)/size(LFP.data,2)); % 确定剩余的样本点如果转换为分段数据，可以分成多少段
%     %     band_phase_LFP = band_phase_LFP(:,1:epoch_num2*size(LFP.data,2)); % 依据可以分成的段数，截取数据
%     %     band_phase_LFP = reshape(band_phase_LFP,[size(LFP.data,2) epoch_num2]);% 将数据重新转换为三维：电极*样本点*分段
%
%
%     % set the sliding window and step size
%     winSize = 30*500; % 30 seconds * 50 sample rate
%     dataLength = length(bandPhase_eeg);
%     overlap = 0.5;
%     stepvector = 1 : overlap*winSize : dataLength; % 50% overlap
%     for a = 1: size(stepvector,2)
%         if (stepvector(a)+winSize-1) > dataLength
%             break;
%         end
%         Wins(a,:) = stepvector(a):(stepvector(a)+winSize-1);
%     end
%
%     %% 计算PLV
%     for i = 1:size(Wins,1)
%         x_phase = squeeze(bandPhase_eeg(1,Wins(i,:)));
%         y_phase = squeeze(bandPhase_LFP(1,Wins(i,:)));
%         rp = x_phase - y_phase;
%         %%% PLV
%         sub_plv(i) = abs(sum(exp(1i*rp))/length(rp)); % 计算某个分段的PLV
%     end
%     plv(m) = squeeze(mean(sub_plv,2)); % 对该被试各个分段的PLV计算平均值；
%     clear Wins
%     clear bandPhase_eeg bandPhase_LFP sub_pli sub_plv Wins
%     %     end
% end
%
% save('D:/Rest_EEG_analysis/BNST_Data/PLV_theta','plv');
