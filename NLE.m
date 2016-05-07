function [Y P] = NLE(X, K, d, t, K2, la, d_pca)
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
    W(i, neighbors(:, i)) = exp(- t ./ distance(i, neighbors(:, i)));%exp(-distance(i, neighbors(:, i)) / t);
    W(neighbors(:, i), i) = W(i, neighbors(:, i));
end

W2 = exp(- la ./ distance);%ones(N, N) * la;
% neighbors1 = index(end - K2 : end, :);
neighbors1 = index(1 : K + K2, :);
for i = 1 : N
    W2(i, neighbors1(:, i)) = 0;%1- exp(-distance(i, neighbors1(:, i)) / la);
    W2(neighbors1(:, i), i) = W2(i, neighbors1(:, i));
end
% W2 = W2 * (max(abs(W) / max(abs(W2))));

D = spdiags(sum(W, 2),0,speye(size(W,1)));
L = D - W;

D2 = spdiags(sum(W2, 2),0,speye(size(W2,1)));
L2 = D2 - W2;

Wpca = PCA(X, d_pca);
% L = (Y * L * Y')^-1 * (Y * L2 * Y');
L = (Wpca' * X * L2 * X' * Wpca)^-1 * (Wpca' * X * L * X' * Wpca);
% L = (X * L2 * X')^-1 * (X * L * X');
opts.disp = 0;
opts.isreal = 1; 
opts.issym = 1; 
[P, Lambda] = eigs(L, d + 1, 0, opts);
% [P, Lambda] = eigs(L, d);
% [Y, ~] = eigs(L, d);
%Y = Y(:, 1 : d);
P = real(P);
P = P(:, 2 : d + 1);
P = Wpca * P;
Y = P' * X;
