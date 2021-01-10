%%
clear all
close all
clc


%% DATA LOADING
% Loads raw data in .mat, defines variables and creates a matrix 
% with channels or electrodes (raw) and samples (columns).

%
% Loading of electrophysiological signals from 16 channels 
%

load('Rec_10_Sevo1_LightSt');


%
% DEFINE VARIABLES
%

% Sample frequency in [Hz]
Fs = tSpikyExport.CH01KHz*1000;

% Beginning of recording (time) 
TimeBegin = tSpikyExport.CH01_TimeBegin;

% Time stamps of onset of stimuli (vector)
TTL = tSpikyExport.TTL01_Up;


%
% MATRIX CREATION
%

% It creates a matrix "channels X samples" and moltiplies 1000000 
% to convert the voltage in the order of [uV]


EEG_raw = zeros (16,length(tSpikyExport.CH01));
for kk=1:16
    if kk<10
        EEG_raw(kk,:)=tSpikyExport.(['CH0' num2str(kk)]) .*1000000;
    else
        EEG_raw(kk,:)=tSpikyExport.(['CH' num2str(kk)]) .*1000000;
    end
end
clear kk


%% CHANNEL REJECTION - VISUAL INSPECTION
% Electrophysiological traces from all channels are visually inspected 
% for artefacts or noise. The affected channels are classified 
% as "bad channels" and removed from analysis


%
% DOWNSAMPLING
%

% Electrophysiological traces are band pass filtered and downsampled
% to allow visualization (this operation is not permanent)

r = 100; % decimation factor

 EEG_check_ch = [];
for kk=1:16
       EEG_check_ch(kk,:) = decimate (EEG_raw(kk,:),r);
end
clear kk r


%
% PLOT OF SIGNAL FROM ALL CHANNELS
%

clf
figure (1);

d= 10; % fraction to plot
x1 = 1; % [s]
x2 =  size(EEG_check_ch,2)/d; % [s]
y1 = -500; % [uV]
y2 =  500; % [uV]

subplot (8,2,1);
    plot ( EEG_check_ch(1,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 1')
subplot (8,2,2);
    plot ( EEG_check_ch(2,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 2')
subplot (8,2,3);
    plot ( EEG_check_ch(3,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 3')
subplot (8,2,4);
    plot ( EEG_check_ch(4,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 4')   
subplot (8,2,5);
    plot ( EEG_check_ch(5,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 5')  
subplot (8,2,6);
    plot ( EEG_check_ch(6,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 6')
subplot (8,2,7);
    plot ( EEG_check_ch(7,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 7')
subplot (8,2,8);
    plot ( EEG_check_ch(8,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 8')
subplot (8,2,9);
    plot ( EEG_check_ch(9,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 9')
subplot (8,2,10);
    plot ( EEG_check_ch(10,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 10')
subplot (8,2,11);
    plot ( EEG_check_ch(11,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 11')
subplot (8,2,12);
    plot ( EEG_check_ch(12,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 12')
subplot (8,2,13);
    plot ( EEG_check_ch(13,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 13')
subplot (8,2,14);
    plot ( EEG_check_ch(14,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 14')
subplot (8,2,15);
    plot ( EEG_check_ch(15,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 15')
subplot (8,2,16);
    plot ( EEG_check_ch(16,:),'k') 
    axis([x1,x2, y1,y2])
    xlabel ('[samples]')
    ylabel ('[uV]')
    title  ('Channel 16')

clear x1 x2 y1 y2 d

    
%% CHANNEL REJECTION - MANUAL REMOVAL
% Electrophysiological traces from all channels are visually inspected 
% for artefacts or noise. The affected channels are classified 
% as "bad channels" and removed from analysis


%
% REMOVAL OF BAD CHANNELS 
%

% Define the vector of channels to be removed (0 if no bad channels detected)
badCh = [3]; 
% Define the vector of channels to maintain
EEG_raw = EEG_raw ([1,2,4,5,6,7,8,9,10,11,12,13,14,15,16],:);

clear EEG_check_ch Fs_check_ch Time_check_ch


%% REFERENCING
% 2 parallel referencing procedures are implemented. 
% i) Common average referencing: the average across all channels is computed. 
% The resulting mean trace displays the common noise across channels 
% and is subtracted from the recordings of all channels. 
% This procedure is applied for the analysis of evoked related potentials (e.g. PCI). 
% ii) Bipolar referencing: the electrophysiological recording from 
% the electrode positioned over the right secondary visual cortex (V2) 
% is subtracted from the electrode positioned over the right secondary motor cortex (M2). 
% The resulting fronto-occipital derivation is used to quantify spontaneous activity. 


%
% COMMON AVERAGE REFERECING
%

% Computes the common average across all channels
 Average_ch = mean (EEG_raw,1); 
% Common average subtraction
 EEG_avref = zeros (size(EEG_raw,1),size(EEG_raw,2));
  for kk=1:size(EEG_raw,1)
        EEG_avref(kk,:) = EEG_raw (kk,:) - Average_ch;  
  end
  clear kk Average_ch
 
  
%
% BIPOLAR REFERECING
%

% Define the poles of the derivation
 ch_plus = 1; % channel in M2
 ch_minus= 14; % channel in V2 (to be subtracted)
% Compute the subtraction
 eeg_diff = squeeze (EEG_raw(ch_plus,:)-EEG_raw(ch_minus,:));
 

%% REMOVAL OF STIMULUS ARTEFACT
% The intracranial electrical stimulation normally introduces 
% an artefactual fast and high voltage transient 
% that needs to be removed from electrophysiological recording before filtering. 
% The exact samples corresponding to the onset of all the stimulations are identified 
% and 0.005 s of the signal from those time points are removed and interpolated 
% with a spline function in all channels.


%
% DEFINES PARAMETERS
%

% Finds stimului in samples
triggers = round((TTL-TimeBegin).*Fs);
% Vector of samples
S = (1:size(EEG_avref,2));
% Time range to interpolate from stimulus onset in [s]
Pre_arct_x1 = 0.000;
Post_arct_x2 = 0.005;
% Ammount of the signal used for fitting in [s]
Data_interp = 0.01;      


%
% ARTEFACT REMOVAL AND INTERPOLATION
%

% On signal with common average referecing

        EEG_no_arct = EEG_avref;
    for jj=1:size(EEG_avref,1);
    for kk = 1:length(triggers);
        
        Arct_range = (triggers(kk)+(Pre_arct_x1 * Fs (+1))):(triggers(kk)+(Post_arct_x2 * Fs (+1))); % in sample
        Pre = (triggers(kk)-(Data_interp * Fs)):(triggers(kk)+(Pre_arct_x1 * Fs (+1))-1);
        Post = (triggers(kk)+(Post_arct_x2 * Fs (+1))+1):(triggers(kk)+(Data_interp * Fs));
       
        EEG_no_arct (jj,Arct_range) = interp1(S([Pre Post]), EEG_no_arct(jj,[Pre Post]),Arct_range,'spline');
        
    end
    end    
    

% On signal with bipolar referecing

    for kk = 1:length(triggers);
        
        Arct_range = (triggers(kk)+(Pre_arct_x1 * Fs (+1))):(triggers(kk)+(Post_arct_x2 * Fs (+1))); % in sample
        Pre = (triggers(kk)-(Data_interp * Fs)):(triggers(kk)+(Pre_arct_x1 * Fs (+1))-1);
        Post = (triggers(kk)+(Post_arct_x2 * Fs (+1))+1):(triggers(kk)+(Data_interp * Fs));
       
        eeg_diff (Arct_range) = interp1(S([Pre Post]), eeg_diff ([Pre Post]),Arct_range,'spline');
        
    end
    
    clear S jj kk Arct_range Pre_arct_x1 Post_arct_x2 Data_interp Pre Post trigger
  

%% FILTERING and DOWN-SAMPLING
% Electrophysiological traces from all channels are band pass filtered 
% from 0.5 to 80 Hz with a 3th order Butterworth filter and 
% down-sampled to 500 Hz


%
% OPERATION ON SIGNAL WITH COMMON AVERAGE REFERECING
%
    
% Setting of Butterwoth filter     

  w2=2*80/Fs;  %  2*cutting frequency of low pass/sampling frequency;
            w=[w2];
            [B,A]=butter(3,w,'low');  % 3rd order low pass filter
  w2=2*0.5/Fs; %  2*cutting frequency of high pass/sampling frequency;
             w=[w2];
             [C,D]=butter(3,w,'high'); % 3rd order high pass filter
 
% Performs the filtering   

  for kk=1:size(EEG_no_arct,1)  
      EEG_A(kk,:)=filtfilt(B,A,EEG_no_arct(kk,:));
      EEG_B(kk,:)=filtfilt(C,D, EEG_A(kk,:));
  end
  clear A B C D w w2 kk EEG_A EEG_a EEG_ds

% Performs the down-sampling   

  Fs_new = 500; % [Hz]
  r = Fs/Fs_new; % decimation factor

  EEG_BPf = [];
  for kk=1:size(EEG_B,1)
      EEG_BPf(kk,:) = downsample (EEG_B(kk,:),r); 
  end
  clear kk r EEG_B

  
%
% OPERATION ON SIGNAL WITH BIPOLAR REFERECING
%
    
% Setting of Butterwoth filter     
 
  w2=2*80/Fs;  %  2*cutting frequency of low pass/sampling frequency;
            w=[w2];
            [B,A]=butter(3,w,'low');  % 3rd order low pass filter
  w2=2*0.5/Fs; %  2*cutting frequency of high pass/sampling frequency;
             w=[w2];
             [C,D]=butter(3,w,'high'); % 3rd order high pass filter

% Performs the filtering  

  for kk=1:size(eeg_diff,1) 
      eeg_diff_a(kk,:)=filtfilt(B,A,eeg_diff(kk,:));
      eeg_diff_b(kk,:)=filtfilt(C,D, eeg_diff_a(kk,:));        
  end
  clear A B C D w w2 kk eeg_diff_a

% Performs the down-sampling   

  Fs_new = 500; % [Hz]
  r = Fs/Fs_new; % decimation factor

   EEGbip_BPf = [];
   for kk=1:size(eeg_diff,1)
       EEGbip_BPf (kk,:) = downsample (eeg_diff_b(kk,:),r);
   end
   clear kk r eeg_diff_b eeg_diff
    

%% SINGLE TRIALS EXTRACTION and OFFSET REMOVAL
% The exact samples corresponding to the onset of all the stimulations are identified 
% and evoked related potential (ERP) epochs or trials from -5 to 5 s centred at the stimulus onset (0 s) 
% are extracted for each channel. A matrix samples X channels X trials is created. 
% All epochs from all channels are offset corrected by subtracting the average voltage of their respective baseline (from -1 to 0 s). 

%
% DEFINES PARAMETERS
%

% Defines stimulus onsets in samples with new sampling frequency:
triggers1 = round((TTL-TimeBegin).*Fs_new); 
% Pre to post stimulus time range to extract in [s]
pre_stim = -5;  
post_stim = 5;
Prestim = abs(pre_stim);
% time range used to compute the offset for each trial
Prestim_offset_x1 = -1;  
Prestim_offset_x2 = 0;


%
% OPERATIONS ON SIGNAL WITH COMMON AVERAGE REFERENCING
%

% Trial extraction:
  for jj = 1:size(EEG_no_arct,1)
    a = EEG_BPf (jj,:);
   for kk = 1:length(triggers1)
    eeg (:,jj, kk) = a (triggers1(kk)+(pre_stim * Fs_new (+1)):triggers1(kk)+(post_stim * Fs_new (+1))); 
   end
  end 
clear jj kk a

% Time vector for trials:
    Time1 = (((1:size(eeg,1))-1)./Fs_new)-post_stim; 

% Offset computation:
    offset_matrix = eeg(((Fs_new.*Prestim_offset_x1)+(Fs_new.*Prestim)+1):((Fs_new.*Prestim_offset_x2)+(Fs_new.*Prestim)+1),:,:);
    offset = mean (offset_matrix,1);

% Offset subtraction:
    for kk=1:length(triggers)
        for jj = 1:size(EEG_no_arct,1);
         eeg(:,jj,kk) = eeg(:,jj,kk) - offset(:,jj,kk);
        end
    end
clear kk jj offset_matrix offset 



%
% OPERATIONS ON SIGNAL WITH BIPOLAR REFERENCING
%
 
% Trial extraction:
   for kk = 1:length(triggers1)
    eeg_diff2 (:,kk) = EEGbip_BPf (triggers1(kk)+(pre_stim * Fs_new (+1)):triggers1(kk)+(post_stim * Fs_new (+1)));
   end
clear eeg_diff kk a

% Offset computation:
    offset_matrix_diff = eeg_diff2(((Fs_new.*Prestim_offset_x1)+(Fs_new.*Prestim)+1):((Fs_new.*Prestim_offset_x2)+(Fs_new.*Prestim)+1),:);
    offset_diff = mean (offset_matrix_diff,1);
    
% Offset subtraction:
    for kk=1:length(triggers1)
         eeg_diff2(:,kk) = eeg_diff2(:,kk) - offset_diff(:,kk);
    end
clear kk Prestim_offset_x1 Prestim_offset_x2 offset_matrix offset offset_matrix_diff offset_diff


%% TRIAL REJECTION 
% Trials with high voltage artefacts in their baseline are automatically removed. 
% Threshold for rejection was set to the averaged root mean square 
% of baseline (rms, from -1 to 0 s) across trials + 3 standard deviations. 


%
% DEFINES PARAMETERS FOR THRESHOLD
%

% Time range of baseline used to compute threshold [s]
Prestim_rms_x1 = -1;        
Prestim_rms_x2 = 0; 
% Time range of baseline used to compare [s]
Dependent_rms_x1 = -1;     
Dependent_rms_x2 = 0;
% Computes the root mean square of baseline ranges
Prestim_matrix = eeg (((Fs_new.*Prestim_rms_x1)+(Fs_new.*Prestim)+1):((Fs_new.*Prestim_rms_x2)+(Fs_new.*Prestim)+1),:,:);
Dependent_matrix = eeg (((Fs_new.*Dependent_rms_x1)+(Fs_new.*Prestim)+1):((Fs_new.*Dependent_rms_x2)+(Fs_new.*Prestim)+1),:,:);
RMS_pre = rms (Prestim_matrix, 1); 
RMS_pre_dependent = rms (Dependent_matrix, 1);
clear Prestim_matrix Prestim_rms_x1 Prestim_rms_x2 Dependent_matrix Dependent_rms_x1 Dependent_rms_x2


%
% DEFINES THRESHOLD FOR REJECTION
%

% Mean and standard deviation of RMS from all baselines from al trials:
Mean_RMS_pre = mean (RMS_pre,3);
STD_RMS_pre = std (RMS_pre,0,3);
% threshold at 3 standard deviations:
Treshold = Mean_RMS_pre + (STD_RMS_pre .* (3));


%
% PERFORMS REJECTIONS AND CONSERVES TRIALS WITH STABLE BASELINE
%

% Identifies only those trials having baseline RMS below threshold in all channels 
RMS_pre_dependent = squeeze(RMS_pre_dependent);
for kk = 1:size(RMS_pre_dependent,2);
    for jj = 1:size(RMS_pre_dependent,1);
        A1(jj,kk) = RMS_pre_dependent(jj,kk) < Treshold(jj);
        A2 = mean (A1,1) == 1;
        F = find (A2);
    end
end
clear jj A1 A2 Treshold RMS_pre RMS_pre_dependent Mean_RMS_pre STD_RMS_pre


% Extract only trials with baseline RMS below threshold in all channels 
% in electrophysiological signal with common average referencing 
    eeg_good = eeg(:,:,F); 
 % in electrophysiological signal with bipolar referencing    
    eeg_diff2 = eeg_diff2(:,F);    

clear F


%% CLEAR AND RENAME:

% Clear Variables
clear EEG_avref EEG_BPf EEG_no_arct EEG_raw EEG_ds eeg eeg2 TTL
clear Fs kk sws post_stim pre_stim Poststim Prestim Time2 TimeBegin
clear triggers triggers1 triggers2 tSpikyExport eeg_no_bpf EEGbip_BPf

% Rename Variables
fs = Fs_new;
clear Fs_new
time = Time1;
clear Time1
eeg_diff = eeg_diff2(:,1:90); % conserves the first 90 trials
eeg = eeg_good (:,:,1:90);
clear eeg_diff2 eeg_good


%% SAVING
% The first n = 90 consecutive trials of pre-processed signal are used for analysis and saved. 
% A file .mat with the following variables is created:
% i)  badCh = identify a vector with the number of eventual ?bad channels? referring to the spatial anchoring of the electrodes (below). 
%     If badCh = 0, no ?bad channels? are identified. 
% ii) ch_minus & ch_plus = identify the number of channels associated to the fronto-occipital derivation 
%     used for analysing spontaneous activity (right V2 and right M2 respectively).
% iii)eeg = is the matrix samples X channels X trials that contains data for analysis of ERPs (common average referencing)
% iv) eeg_diff = is the matrix samples X trials that contains data for analysis of spontaneous activity (fronto-occipital bipolar referencing)
% v)  fs = is the sample frequency of the data (after down-sampling)
% vi) time = is a vector representing time samples in seconds for all trial epochs (from -5 to 5 s centred at the stimulus onset, 0 s).


% Save
save ('j07_r1_Light_Sevo1_PREPROCESSED_HBP')
