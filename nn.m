
function [ data, weights ] = nn(INPUTS, OUTPUTS, ...
    TEST_INPUTS, TEST_OUTPUTS, weights, HIDDEN_LAYERS, EPSILON, ...
    DELTA_LEARN, ALPHA, LAMBDA, PROB_OF_DROPOUT, NUM_OF_EPOCHS, ...
    STRIDE, STRIDE_PER_FILE)


NUM_OF_ITERATIONS = floor(NUM_OF_EPOCHS * (length(INPUTS)/STRIDE));
% Data is captured every EPOCH.
DATA_CAPTURE_RATE = floor(length(INPUTS)/STRIDE);
epsilon = EPSILON; % Learning Rate
alpha = ALPHA;  % Momentum
NUM_OF_OUTPUTS = STRIDE;

data = zeros(DATA_CAPTURE_RATE, 2);
delta = zeros(NUM_OF_OUTPUTS, 1);

i=0;
error = 0;

for prevWeights = 0:length(weights)-1
    prevDeltaWeights{prevWeights+1} = zeros(size(weights{end-prevWeights}));    
end

for a = 1:NUM_OF_ITERATIONS
    
    if(mod(a, 50) == 0)
        X = sprintf('Iteration %d of %d.\n',a,NUM_OF_ITERATIONS);
        fprintf(X)
    end
    
    % Store Error/Cost for later plotting
    if (mod(a, DATA_CAPTURE_RATE) == 0)
        reduction = epsilon * DELTA_LEARN;
        epsilon = epsilon - reduction;
        
        i = i + 1;
        
        cost = 0.5 * sum(errors);
        error = 0;
%         disp(cost)
        data(i, 1) = i;
        data(i, 2) = cost;
        data(i, 3) = evaluateTrainingData( TEST_INPUTS, TEST_OUTPUTS, STRIDE, ...
            PROB_OF_DROPOUT, weights);
        %         disp(cost);
    end
    
    
    % Iterate over the inputs
    index = mod(a, STRIDE_PER_FILE) + 1;
    current_pos = (index-1)*STRIDE;
    input = INPUTS(current_pos+1:current_pos+(STRIDE));
    
    % FeedForward through the network
    
    tempWeights = weights;
    [ tempWeights ] = dropout( tempWeights, PROB_OF_DROPOUT );
    
    [ hidden ] = feedforward( input, tempWeights);
    desired_output = OUTPUTS(current_pos+1:current_pos+(STRIDE));
    
    % Calculate Error/Cost for the current training instance
    for g = 1:NUM_OF_OUTPUTS
        error = error + (desired_output(g,1) - hidden{end}(g,1))^2;
    end
    errors(index, 1) = error;
    
    % Backpropagate
    
    % Backpropagate the output layer
    
    [ weights{end}, delta, prevDeltaWeights{1} ] = backpropagateOutputLayer( hidden{end}, ...
        desired_output, hidden{end-1}, tempWeights{end}, prevDeltaWeights{1},  epsilon, alpha, ...
        LAMBDA, NUM_OF_ITERATIONS, NUM_OF_OUTPUTS, HIDDEN_LAYERS(end));    
    
    % Backpropagate the Hidden Layers
    [ weights{end-1}, prevDeltaWeights{2}, newDelta ] = backpropagateHiddenLayer(  hidden{end-2},...
        hidden{end-1}, tempWeights{end-1}, tempWeights{end}, delta, prevDeltaWeights{2}, epsilon, alpha, ...
        LAMBDA, NUM_OF_ITERATIONS, NUM_OF_OUTPUTS, HIDDEN_LAYERS(end), HIDDEN_LAYERS(2));
    
    [ weights{end-2}, prevDeltaWeights{3}, newDelta ] = backpropagateHiddenLayer(  hidden{end-3},...
        hidden{end-2}, tempWeights{end-2}, tempWeights{end-1}, newDelta, prevDeltaWeights{3}, epsilon, alpha, ...
        LAMBDA, NUM_OF_ITERATIONS, HIDDEN_LAYERS(3), HIDDEN_LAYERS(2), HIDDEN_LAYERS(1));
    
    [ weights{end-3}, prevDeltaWeights{4}, newDelta ] = backpropagateHiddenLayer(  hidden{end-4},...
        hidden{end-3}, tempWeights{end-3}, tempWeights{end-2}, newDelta, prevDeltaWeights{4}, epsilon, alpha, ...
        LAMBDA, NUM_OF_ITERATIONS, HIDDEN_LAYERS(2), HIDDEN_LAYERS(1), STRIDE);
    
    
    
    
    
end

end
