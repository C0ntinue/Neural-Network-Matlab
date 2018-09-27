
function y = evaluateNetwork( data, INPUTS, CLEAN, STRIDE, weights)


figure(1)
x = data(:,1);
y = data(:,2);
z = data(:,3);
y = log(y);
z = log(z);
plot(x,y);
hold
plot(x, z, 'r');
legend('Training Data','Test Data')
ylabel('Cost')
xlabel('Iteration')


NUM_OF_ITERATIONS = floor(length(INPUTS)/STRIDE);
y = 0;

for a = 1:NUM_OF_ITERATIONS
    
    % Iterate over the inputs
    index = mod(a, NUM_OF_ITERATIONS+1);
    current_pos = (index-1)*STRIDE;
    input = INPUTS(current_pos+1:current_pos+(STRIDE));
    
    % FeedForward through the network
    [network_output] = feedforward(input, weights); 
    
    y = cat(1, y, network_output{end});
    
end

figure(2)
plot(y, 'r');
hold on
plot(CLEAN);
legend('Output From Network', 'Clean Waveform')
hold off
end