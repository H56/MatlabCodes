function Y = LE(X, K, d, t)
[D, N] = size(X);
Delta = 0;
if(K > D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  Delta = 1e-1; % regularlizer in case constrained fits are ill conditioned
end
%STEP1: compute the neighbors
X2 = sum(X .^ 2, 1);
distance = sqrt(bsxfun(@plus, X2, bsxfun(@minus, X2', 2 * (X' * X))));
[~, index] = sort(distance, 1);
neighbors = index(2 : K + 1, :);

W = sparse(N, N);
for i = 1 : N
    W(i, neighbors(:, i)) = exp(-distance(i, neighbors(:, i)) / t);
    W(neighbors(:, i), i) = exp(-distance(i, neighbors(:, i)) / t);
end

D = spdiags(sum(W, 2),0,speye(size(W,1)));
L = D - W;

opts.tol = 1e-9;
opts.issym=1; 
opts.disp = 5; 
% [Y,V] = eigs(D^(-1) * L + 10*eps*speye, d, 'sm', opts);
[Y, ~] = eigs(L, d + 1, 'sm', opts);
Y = Y(:, 1 : d);
k1 = ones(1, length(D)) * D * ones(length(D), 1);
Y = bsxfun(@minus, Y, (Y' * D * ones(length(Y), 1))' / k1);
Y = bsxfun(@rdivide, Y, sqrt(diag(Y' * D * Y)'));
Y = Y(:, d : -1 : 1);
Y = Y';
% Y = Y(:, end : -1 : 1);