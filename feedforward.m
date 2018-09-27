function [ output ] = feedforward( input, weights)
   output = cell(length(weights)+1, 1);
   output{1} = input;
   input(end+1, 1)= 1; % Bias added
   
   for i = 1:length(weights)
      
       temp = (weights{i} * input);
       temp = activation_rev2(temp, 'false');
       output{i+1} = temp;
       
       input = temp;
       input(end+1, 1)= 1; % Bias added
   end
end