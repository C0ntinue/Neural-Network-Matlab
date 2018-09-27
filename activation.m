function  [y] = activation(x, derivative)

if (strcmp(derivative, 'true'))
    y = 1 - (tanh(x)).^2;
else
    y = tanh(x);
end