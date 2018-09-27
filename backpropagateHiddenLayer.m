function [ weights, prevDeltaWeights, newDelta ] = backpropagateHiddenLayer(  input,...
    hidden, weights, Weights2, delta, prevDeltaWeights, epsilon, alpha, LAMBDA, n, ...
    NUM_OF_OUTPUTS, NUM_OF_HIDDEN_UNITS, NUM_OF_INPUTS)

currentDeltaWeights = zeros(NUM_OF_HIDDEN_UNITS, NUM_OF_INPUTS + 1);
newDelta = zeros(NUM_OF_HIDDEN_UNITS, 1);

for d = 1:NUM_OF_HIDDEN_UNITS
    
    dEtotal = 0;
    
    for e = 1:NUM_OF_OUTPUTS
        dEe = delta(e, 1) * (Weights2(e, d));
        dEtotal = dEtotal + dEe;
    end
    
    doutdnet2 = activation(hidden(d, 1), 'true');
    newDelta(d, 1) = dEtotal * doutdnet2;
    
    for f = 1:NUM_OF_INPUTS
        dnetdwf = input(f, 1);
        dEdwf = dEtotal * doutdnet2 * dnetdwf;
        
        currentDeltaWeights(d,f) = -(epsilon * dEdwf);
        
        weights(d,f) = weights(d,f) + currentDeltaWeights(d,f) ...
            + (alpha * prevDeltaWeights(d,f)) ...
            + (currentDeltaWeights(d,f) * epsilon * LAMBDA) / n;
        
        prevDeltaWeights(d,f) = currentDeltaWeights(d,f);
    end
    
    dEdbias = dEtotal * doutdnet2 * 1;
    currentDeltaWeights(d,NUM_OF_INPUTS+1) = -(epsilon * dEdbias);
    
    weights(d,NUM_OF_INPUTS+1) = ...
        weights(d,NUM_OF_INPUTS+1) + currentDeltaWeights(d,NUM_OF_INPUTS+1) ...
        + (alpha * prevDeltaWeights(d,NUM_OF_INPUTS+1));
    
    prevDeltaWeights(d,NUM_OF_INPUTS+1) = currentDeltaWeights(d,NUM_OF_INPUTS+1);
    
end


end