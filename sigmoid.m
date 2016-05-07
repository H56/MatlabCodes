function ret = sigmoid(x, y, k, s)
ret = 1 ./ (exp(k * (x' * y) + s));