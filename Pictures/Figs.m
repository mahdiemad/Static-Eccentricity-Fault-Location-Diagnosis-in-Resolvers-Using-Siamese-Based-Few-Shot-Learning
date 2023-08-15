%% data read
clc
clear all
Df = xlsread('../CompleteData.xlsx',1);
%% preprocess
Frequency = Df(1:end,2:76);
Frequency(1,:)=(Frequency(1,:)+1)*8;
log_Frequency=abs(normalize(log(Frequency(2:end,:)),'zscore'));
norm_log_Frequency=(log_Frequency-min(log_Frequency()))./(max(log_Frequency)-min(log_Frequency));

%% plot
figure(1);
plot(Frequency(1,:),Frequency(43,:),'-','LineWidth',1.5,'Color','b');
xlim([0 400]);
xticks([0:40:400]);

% log normalized frequency
figure(2);
plot(Frequency(1,:),norm_log_Frequency(43,:),'-','LineWidth',1.5,'Color','b');
xlim([0 400]);
xticks([0:40:400]);
%% color map
load('test33.mat');
load('FOPcolormap.mat');
sample=squeeze(arr(6,:,:));
imagesc(sample);
colormap(FOPcolormap); 
colorbar
% axis off
xticks([0:5:75]);
yticks([0:5:75]);
ax = gca;
tick_scale_factor = 8;
ax.XTickLabel = ax.XTick * tick_scale_factor;
ax.YTickLabel = ax.YTick * tick_scale_factor;
ax.ZTickLabel = ax.ZTick * tick_scale_factor;

%% Confusion matrix
confusion_data= xlsread('confusion.xlsx',1);
CM=confusionmat(confusion_data(:,2),confusion_data(:,3));
classLabels=categorical({'Fault 1','Fault 2','Fault 3','Fault 4'});
heatmap(classLabels,classLabels,CM);

%% Loss and Accuracy

