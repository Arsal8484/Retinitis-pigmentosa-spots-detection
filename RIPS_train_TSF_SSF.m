
close all
clear all
% load 'lgraph_inConv_stride' %% previous successful training 
load 'LG_threeFuse_9Dec2020' %% 09 December 2020 three fusion and Dilation factor, BN, ReLU input side
   lgraph=lgraph_1;
% figure,plot(lgraph)
  %%%%%%%%%%%%%%%%%%%%%%% outer identity paths start 
  
% % lgraph = removeLayers(lgraph,'inputImage'); % Removing input image layer to change the image size
%   inputLayer = imageInputLayer([480 720 3],'Name','inputLayer'); % new input image layer
%   lgraph = replaceLayer(lgraph,'inputLayer',inputLayer);
% % lgraph = addLayers(lgraph, inputImage); % adding new layer with weights
% % lgraph = connectLayers(lgraph,'inputImage','conv1_1'); 
%   
%   
%   lgraph = disconnectLayers(lgraph, 'decoder1_relu_2' ,'decoder1_conv1');
%   OS1 = additionLayer(2,'Name','OS1');
%   lgraph = addLayers(lgraph, OS1);
%   lgraph = connectLayers(lgraph, 'relu1_1' ,'OS1/in1');
%   lgraph = connectLayers(lgraph, 'decoder1_relu_2' ,'OS1/in2');
%   lgraph = connectLayers(lgraph, 'OS1' ,'decoder1_conv1');
%   
%   
%   lgraph = disconnectLayers(lgraph, 'decoder2_relu_2' ,'decoder2_conv1');
%   OS2 = additionLayer(2,'Name','OS2');
%   lgraph = addLayers(lgraph, OS2);
%   lgraph = connectLayers(lgraph, 'relu2_1' ,'OS2/in1');
%   lgraph = connectLayers(lgraph, 'decoder2_relu_2' ,'OS2/in2');
%   lgraph = connectLayers(lgraph, 'OS2' ,'decoder2_conv1');
%   
%   
%   
%   lgraph = disconnectLayers(lgraph, 'decoder3_relu_2' ,'decoder3_conv1');
%   OS3 = additionLayer(2,'Name','OS3');
%   lgraph = addLayers(lgraph, OS3);
%   lgraph = connectLayers(lgraph, 'relu3_1' ,'OS3/in1');
%   lgraph = connectLayers(lgraph, 'decoder3_relu_2' ,'OS3/in2');
%   lgraph = connectLayers(lgraph, 'OS3' ,'decoder3_conv1');
%   
%   
%   lgraph = disconnectLayers(lgraph, 'decoder4_relu_2' ,'decoder4_conv1');
%   OS4 = additionLayer(2,'Name','OS4');
%   lgraph = addLayers(lgraph, OS4);
%   lgraph = connectLayers(lgraph, 'relu4_1' ,'OS4/in1');
%   lgraph = connectLayers(lgraph, 'decoder4_relu_2' ,'OS4/in2');
%   lgraph = connectLayers(lgraph, 'OS4' ,'decoder4_conv1');
%   
  %%%%%%%%%%%%%%%%%%%%%%% outer identity paths end 
  
%   figure,plot(lgraph)
%   load 'prath_lgraph'
%  figure,plot(lgraph)
 
 
Folder = 'J:\9th\RIPS_Experiment\Fold1';% Main directory to all images
 
train_img_dir = fullfile(Folder,'Train');%Training image directory
imds = imageDatastore(train_img_dir); 
 

classes = ["P","NonP"]; %% Class names
labelIDs   = [1 0]; % Class id


train_label_dir = fullfile(Folder,'Train GT');  %% Training label directory
pxds = pixelLabelDatastore(train_label_dir,classes,labelIDs);

tbl = countEachLabel(pxds); % occurance of iris and non-iris pixels


frequency = tbl.PixelCount/sum(tbl.PixelCount); % frequency of each class

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;    % frequency balancing median 

 pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights); % adding weights tp pixel classification layer


lgraph = removeLayers(lgraph,'labels'); % deleting previous layer
lgraph = addLayers(lgraph, pxLayer); % adding new layer with weights
lgraph = connectLayers(lgraph,'softmax','labels');% retreiving the connection

%  figure,plot(lgraph)

%%% Training options %%%%%

options = trainingOptions('adam', ...
    'SquaredGradientDecayFactor',0.95, ...
    'GradientThreshold',7, ...
    'GradientThresholdMethod','global-l2norm', ...
    'Epsilon',1e-5, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0008, ...
    'MaxEpochs',15, ...  
    'MiniBatchSize',6, ...
    'CheckpointPath',tempdir, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',2);

augment_data = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-5 5],'RandYTranslation',[-5 5]); % optional data augmentation


training_data = pixelLabelImageDatastore(imds,pxds,...
    'DataAugmentation',augment_data); %% complete image+label data


[net, info] = trainNetwork(training_data,lgraph,options);% Train the network