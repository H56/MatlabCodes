function [index, distance] = KNNIndex(X, k)    
X2 = sum(X .^ 2, 1);
distance = sqrt(bsxfun(@plus, X2, bsxfun(@minus, X2', 2 * (X' * X))));
[~, index] = sort(distance, 1);
index = index(2 : k + 1, :);
distance = distance(1 : k + 1, :);
