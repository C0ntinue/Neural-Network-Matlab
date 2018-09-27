function [ weightsOutput, delta, prevDeltaWeights ] = backpropagateOutputLayer( network_output, ...
    desired_output, hidden, weightsOutput, prevDeltaWeights, epsilon, alpha, LAMBDA, n,...
    NUM_OF_OUTPUTS, NUM_OF_HIDDEN_UNITS)

currentDeltaWeights = zeros(NUM_OF_OUTPUTS, NUM_OF_HIDDEN_UNITS + 1);
delta = zeros(NUM_OF_OUTPUTS, 1);

for b = 1:NUM_OF_OUTPUTS
    
    output = network_output(b, 1);
    dEdout = -(desired_output(b, 1) - output);
    doutdnet = activation(output, 'true');
    delta(b, 1) = dEdout * doutdnet;
    
    % Update each connection to the output unit.
    for c = 1:NUM_OF_HIDDEN_UNITS
        dnetdwc = hidden(c, 1);
        dEdwc = delta(b, 1) * dnetdwc;
        currentDeltaWeights(b,c) = -(epsilon * dEdwc);
        
        weightsOutput(b,c) = weightsOutput(b,c) + currentDeltaWeights(b,c) ...
            + (alpha * prevDeltaWeights(b,c)) ...
            + (currentDeltaWeights(b,c) * epsilon * LAMBDA) / n;
        
        prevDeltaWeights(b,c) = currentDeltaWeights(b,c);
    end
    
    % Update the Biases.
    dEdbias = delta(b, 1) * 1;
    currentDeltaWeights(b,NUM_OF_HIDDEN_UNITS+1) = -(epsilon * dEdbias);
    
    weightsOutput(b, NUM_OF_HIDDEN_UNITS + 1) =  ...
        weightsOutput(b,NUM_OF_HIDDEN_UNITS + 1) ...
        + currentDeltaWeights(b,NUM_OF_HIDDEN_UNITS + 1) ...
        + (alpha * prevDeltaWeights(b,NUM_OF_HIDDEN_UNITS + 1));
    
    prevDeltaWeights(b,NUM_OF_HIDDEN_UNITS+1) = ...
        currentDeltaWeights(b,NUM_OF_HIDDEN_UNITS+1);
end
end