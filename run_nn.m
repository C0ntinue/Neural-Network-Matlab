 
 clear all;
 clc;
 
 
 rng('default');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 [INPUTS,Fs] = audioread('trainingData.WAV');
 [OUTPUTS,Fs] = audioread('clean.WAV');
 [TEST_INPUT,Fs] = audioread('testData.WAV');
 [TEST_OUTPUT,Fs] = audioread('Testclean.WAV');

 STRIDE = 500;
 STRIDE_PER_FILE = 4;
 DELTA_LEARN = 0.0;
 EPSILON = 0.1;
 ALPHA = 0.0;
 LAMBDA = 0.0;
 PROB_OF_DROPOUT = 0.5;
 
 NUM_OF_EPOCHS = 100;
 NUM_OF_IMPULSES = 100;
 IMPULSE_PERCENTAGE = 0.6;
 NUM_OF_TRAINING = 10;
 NUM_OF_TESTING = 1;
 [ INPUTS, OUTPUTS, TEST_INPUTS, TEST_OUTPUTS, Fs] = singlePhoneme( ...
     NUM_OF_TRAINING, NUM_OF_TESTING, NUM_OF_IMPULSES, IMPULSE_PERCENTAGE);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 clear weights
 HIDDEN_LAYERS = [20 30 20];
 % Weights are initilised in the range [1, -1]
 weights{1} = 2 * rand(HIDDEN_LAYERS(1), STRIDE + 1) - 1;
 for i =1:length(HIDDEN_LAYERS)-1
     weights{i+1} = 2 * rand(HIDDEN_LAYERS(i+1), HIDDEN_LAYERS(i) + 1) - 1;
 end
 weights{end+1} = 2 * rand(STRIDE, HIDDEN_LAYERS(end) + 1) - 1;
 
 theWeights = weights;
 tic
 [ data, theWeights ]  = nn(INPUTS, OUTPUTS, ...
     TEST_INPUTS, TEST_OUTPUTS, theWeights, HIDDEN_LAYERS, ...
     EPSILON, DELTA_LEARN, ALPHA, LAMBDA, PROB_OF_DROPOUT, NUM_OF_EPOCHS, ...
     STRIDE, STRIDE_PER_FILE);
 
 time_taken = toc

 save('[20_50_30]_0.1(variable_0.001)_10_0.0m_0.0L_0.0001D_10train-1test.mat')






