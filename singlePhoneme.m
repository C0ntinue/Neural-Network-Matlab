
% Clips Output to be in range of activation function
function [ TRAIN_INPUTS, TRAIN_OUTPUTS, TEST_INPUTS, TEST_OUTPUTS, Fs] = singlePhoneme( ...
    NUM_OF_TRAINING, NUM_OF_TESTING, NUM_OF_IMPULSES, IMPULSE_PERCENTAGE)

rng('default');

filename_in = char('SA1.WAV');

[y,Fs] = audioread(filename_in);

y = y(6642:8641, 1); % eh sound

% sound(y, Fs)
y = -1 + 2.*(y - min(y))./(max(y) - min(y));

AUDIO_LENGTH = size(y,1);
IMPULSE_VALUE = max(y) * IMPULSE_PERCENTAGE;

corruptedFile = 0;
cleanFile = 0;

for j = 1:NUM_OF_TRAINING
    
    noise = zeros(AUDIO_LENGTH, 1);
    for k = 0:NUM_OF_IMPULSES
        spacing = ceil(AUDIO_LENGTH*rand());
        noise(spacing) = IMPULSE_VALUE;
    end
    
    % plot(noise)
    corrupted = y + noise;
    corruptedFile = cat(1, corruptedFile, corrupted);
    cleanFile = cat(1, cleanFile, y);
    
end

corruptedFile(1) = [];
cleanFile(1) = [];

corruptedTestFile = 0;
cleanTestFile = 0;

for j = 1:NUM_OF_TESTING
    
    noise = zeros(AUDIO_LENGTH, 1);
    for k = 0:NUM_OF_IMPULSES
        spacing = ceil(AUDIO_LENGTH*rand());
        noise(spacing) = IMPULSE_VALUE;
    end
    
    corrupted = y + noise;
    corruptedTestFile = cat(1, corruptedTestFile, corrupted);
    cleanTestFile = cat(1, cleanTestFile, y);
    
end

corruptedTestFile(1) = [];
cleanTestFile(1) = [];


corruptedFile(corruptedFile>1) = 1;
cleanFile(cleanFile>1) = 1;
corruptedTestFile(corruptedTestFile>1) = 1;
cleanTestFile(cleanTestFile>1) = 1;


TRAIN_INPUTS = corruptedFile;
TRAIN_OUTPUTS = cleanFile;
TEST_INPUTS = corruptedTestFile;
TEST_OUTPUTS = cleanTestFile;

end
