
function [ weights ] = dropout( weights, PROB_OF_DROPOUT )

rng('shuffle')

for a = 1:length(weights)-1
    
    for b = 1:size(weights{a}, 1)
        probability = rand();
        if(probability<PROB_OF_DROPOUT)
            weights{a}(b, :) = 0;
            weights{a+1}(:, b) = 0;
        else
            continue;
        end
    end
end

end

