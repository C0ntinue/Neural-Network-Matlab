
function y = evaluateTrainingData( INPUTS, CLEAN, STRIDE, PROB_OF_DROPOUT, weights)

NUM_OF_ITERATIONS = floor(length(INPUTS)/STRIDE);
error = 0;
for a = 1:NUM_OF_ITERATIONS
    
    % Iterate over the inputs
    index = mod(a, NUM_OF_ITERATIONS+1);
    current_pos = (index-1)*STRIDE;
    input = INPUTS(current_pos+1:current_pos+(STRIDE));
    desired_output = CLEAN(current_pos+1:current_pos+(STRIDE));
    
    % FeedForward through the network
    [network_output] = feedforward(input, weights);
    
    for b = 1:STRIDE
        
        error = error + (desired_output(b,1) - network_output{end}(b,1))^2;
        
    end
    
end
% plot(desired_output)
% hold
% plot(input, 'r')
% disp('train')
y = 0.5 * error;

end
