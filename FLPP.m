function P = FLPP(X, Ls, K, d, t1, t2)
[D, N] = size(X);
n = length(X);
c = length(Ls);
if K > min(Ls(:, 2) - Ls(:, 1) + 1)
    K = min(Ls(:, 2) - Ls(:, 1) + 1) - 1;
end
mu_i = zeros(n, c);
SB = zeros(length(X));
for i = 1 : c
    num = Ls(i, 2) - Ls(i, 1) + 1;
    Xc = X(:, Ls(i, 1) : Ls(i, 2));
    % c-th class mean
    mu_i(:, i) = mean(Xc, 2);
    
    % compute the L^(c)
    X2 = sum(Xc .^ 2, 1);
    distance = sqrt(abs(bsxfun(@plus, X2, X2') - 2 * (Xc' * Xc)));
    [~, index] = sort(distance, 1);
    neighbors = index(2 : K + 1, :);
    W = sparse(num, num);
    for i = 1 : num
        W(i, neighbors(:, i)) = exp(-distance(i, neighbors(:, i)) / t1);
        W(neighbors(:, i), i) = exp(-distance(i, neighbors(:, i)) / t1);
    end
    D = spdiags(sum(W, 2),0,speye(size(W,1)));
    L = D - W;
    SB = SB + Xc * L * Xc';
end

mu2 = sum(mu_i .^ 2, 1);
mean_d = sqrt(abs(bsxfun(@plus, mu2, mu2') -2 * (mu_i' * mu_i)));
Wmean = zeros(c);
for i = 1 : c
    mean_d(i, i) = 1;
    Wmean(i, :) = exp(-mean_d(i, :) / t2);
    Wmean(i, i) = 0;
end
Dmean = spdiags(sum(Wmean, 2),0,speye(size(Wmean,1)));
Lmean = Dmean - Wmean;
SW = mu_i * Lmean * mu_i';

[~, P] = PCA(X, 20);
[A, dump] = eig(pinv(P * SW * P') * (P * SB * P'));

% [A, dump] = eig(pinv(SW) * SB);
[dump, index] = sort(diag(abs(dump)));
A = A(:, index);
col = find(dump > 0, 1);
F = A(:, col : col + d - 1);
P = P' * F;
