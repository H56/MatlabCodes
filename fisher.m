function [Y, F] = fisher(X, L, d_pca, d)
% X: columns of data
% L: labels, tow dimension, [start, end]
% d_pac: PCA wanted dimension
% d: wanted dimension

% compute means
X = bsxfun(@minus, X, mean(X, 2));
[m, ~] = size(X);
c = length(L);
mu_i = zeros(m, c);
for i = 1 : c
    mu_i(:, i) = mean(X(:, L(i, 1) : L(i, 2)), 2);
end
mu = mean(mu_i, 2);

% compute SB
SB = zeros(m, m);
for i = 1 : c
    SB = SB + (L(i, 2) - L(i, 1) + 1) * (mu_i(:, i) - mu) * (mu_i(:, i) - mu)';
end

% compute SW
SW = zeros(m, m);
for i = 1 : c
    tmp = bsxfun(@minus, X(:, L(i, 1) : L(i, 2)), mu_i(:, i));
    SW = SW + tmp * tmp';
end

% compute PCA
[~, P] = PCA(X, d_pca);

% result
[F, ~] = eigs((P * SW * P')^(-1) * (P * SB * P'), d);
F = P' * F;
Y = F' * X;
