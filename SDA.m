function A = SDA(X, labels, p, a, d)
X = bsxfun(@minus, X, mean(X, 2));
[n, m] = size(X);
len_l = length(labels);

[labels, index] = sort(labels);
X(:, 1 : len_l) = X(:, index);
L = [];
pre = [];
for i = 1 : length(labels)
    if isempty(pre)
        pre = i;
    end
    if labels(pre) ~= labels(i)
        L = [L; [pre, i - 1]];
        pre = i;
    end
end
L = [L; [pre, i]];
c = length(L);
mu_i = zeros(n, c);
for i = 1 : c
    mu_i(:, i) = mean(X(:, L(i, 1) : L(i, 2)), 2);
end
mu = mean(mu_i, 2);

% compute Sb
Sb = zeros(n, n);
for i = 1 : c
    Sb = Sb + (L(i, 2) - L(i, 1) + 1) * (mu_i(:, i) - mu) * (mu_i(:, i) - mu)';
end

% compute St
St = bsxfun(@minus, X, mu);
St = St * St';

[neighbors, distance] = KNNIndex(X, p);
W = sparse(m, m);
for i = 1 : m
    W(i, neighbors(:, i)) = 1;
    W(neighbors(:, i), i) = W(i, neighbors(:, i));
end
D = spdiags(sum(W, 2), 0, speye(size(W,1)));
L = D - W;
tmp = St + a * X * L * X';
if ~isempty(find(sum(abs(tmp)) == 0))
    tmp = tmp + 10^(-9) * eye(size(tmp));    
end
[A, D] = eigs( (tmp)^(-1) * Sb , d);
A = real(A);