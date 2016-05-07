function ret = gaussian_kernel(x, y, delta)
x2 = sum(x .^ 2, 1);
y2 = sum(y .^ 2, 1);
ret = exp(-abs(bsxfun(@plus, x2', y2) - 2 * (x' * y)) / (2 * delta^2));