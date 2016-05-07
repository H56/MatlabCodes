function [Y F] = kernel_PCA(X, K, d)
% X: columns of data
% L: labels, tow dimension, [start, end]
% d_pac: PCA wanted dimension
% d: wanted dimension

% compute means
% K = gaussian_kernel(X, X, 10^7);
% [Apca, D] = eigs(K, [], Dpca);
[F, D] = eig(K);
[D, index] = sort(diag(D), 'descend');
F = F(:, index(1 : d));
Y = F' * K;
